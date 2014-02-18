from nipype.interfaces.matlab import MatlabCommand
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec, traits, InputMultiPath,
                                    File, TraitedSpec, OutputMultiPath)
from nipype.utils.filemanip import split_filename
import os, os.path as op
from string import Template
import shutil
import logging

logging.basicConfig()
iflogger = logging.getLogger('interface')

class SingleSubjectICAInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True,
    desc='The input fMRI data as separate images')
    desired_number_of_components = traits.Int(30, usedefault=True, desc='The desired number of independent components to split the data into.')
    prefix = traits.Str(desc='A prefix for the output files')

class SingleSubjectICAOutputSpec(TraitedSpec):
    mask_image = File(exists=True, desc='ICA mask image')
    parameter_mat_file = File(exists=True, desc='ICA parameter MATLAB .mat file')
    ica_mat_file = File(exists=True, desc='ICA MATLAB .mat file')
    subject_mat_file = File(exists=True, desc='Subject MATLAB .mat file')
    results_log_file = File(exists=True, desc='Results log file')
    independent_component_images = OutputMultiPath(File, exists=True, desc='ICA mask image')
    independent_component_timecourse = File(exists=True, desc='IC timecourse image')

class SingleSubjectICA(BaseInterface):
    """
    Wraps part of the GIFT ICA Toolbox in order to perform independent component analysis on a single subject

    """
    input_spec = SingleSubjectICAInputSpec
    output_spec = SingleSubjectICAOutputSpec

    def _run_interface(self, runtime):
        in_files = self.inputs.in_files
        data_dir = op.join(os.getcwd(),'origdata')
        if not op.exists(data_dir):
            os.makedirs(data_dir)
        all_names = []
        print 'Multiple ({n}) input images detected! Copying to {d}...'.format(n=len(self.inputs.in_files), d=data_dir)
        for in_file in self.inputs.in_files:
            path, name, ext = split_filename(in_file)
            shutil.copyfile(in_file, op.join(data_dir, name) + ext)
            if ext == '.img':
                shutil.copyfile(op.join(path, name) + '.hdr', op.join(data_dir, name) + '.hdr')
            elif ext == '.hdr':
                shutil.copyfile(op.join(path, name) + '.img', op.join(data_dir, name) + '.img')
            all_names.append(name)
        print 'Copied!'

        input_files_as_str = op.join(data_dir, os.path.commonprefix(all_names) + '*' + ext)
        number_of_components = self.inputs.desired_number_of_components
        output_dir = os.getcwd()
        prefix = self.inputs.prefix
        d = dict(output_dir=output_dir, prefix=prefix, number_of_components=number_of_components, in_files=input_files_as_str)
        variables = Template("""

        %% After entering the parameters, use icatb_batch_file_run(inputFile);

        modalityType = 'fMRI';
        which_analysis = 1;
        perfType = 1;
        keyword_designMatrix = 'no';
        dataSelectionMethod = 4;
        input_data_file_patterns = {'$in_files'};
        dummy_scans = 0;
        outputDir = '$output_dir';
        prefix = '$prefix';
        maskFile = [];
        group_pca_type = 'subject specific';
        backReconType = 'gica';

        %% Data Pre-processing options
        % 1 - Remove mean per time point
        % 2 - Remove mean per voxel
        % 3 - Intensity normalization
        % 4 - Variance normalization
        preproc_type = 3;
        pcaType = 1;
        pca_opts.stack_data = 'yes';
        pca_opts.precision = 'single';
        pca_opts.tolerance = 1e-4;
        pca_opts.max_iter = 1000;
        numReductionSteps = 2;
        doEstimation = 0;
        estimation_opts.PC1 = 'mean';
        estimation_opts.PC2 = 'mean';
        estimation_opts.PC3 = 'mean';

        numOfPC1 = $number_of_components;
        numOfPC2 = $number_of_components;
        numOfPC3 = 0;

        %% Scale the Results. Options are 0, 1, 2, 3 and 4
        % 0 - Don't scale
        % 1 - Scale to Percent signal change
        % 2 - Scale to Z scores
        % 3 - Normalize spatial maps using the maximum intensity value and multiply timecourses using the maximum intensity value
        % 4 - Scale timecourses using the maximum intensity value and spatial maps using the standard deviation of timecourses
        scaleType = 0;
        algoType = 1;
        refFunNames = {'Sn(1) right*bf(1)', 'Sn(1) left*bf(1)'};
        refFiles = {which('ref_default_mode.nii'), which('ref_left_visuomotor.nii'), which('ref_right_visuomotor.nii')};
        %% ICA Options - Name by value pairs in a cell array. Options will vary depending on the algorithm. See icatb_icaOptions for more details. Some options are shown below.
        %% Infomax -  {'posact', 'off', 'sphering', 'on', 'bias', 'on', 'extended', 0}
        %% FastICA - {'approach', 'symm', 'g', 'tanh', 'stabilization', 'on'}
        icaOptions =  {'posact', 'off', 'sphering', 'on', 'bias', 'on', 'extended', 0};
        """).substitute(d)

        file = open('input_batch.m', 'w')
        file.writelines(variables)
        file.close()

        script = """param_file = icatb_read_batch_file('input_batch.m');
        load(param_file);
        global FUNCTIONAL_DATA_FILTER;
        global ZIP_IMAGE_FILES;
        FUNCTIONAL_DATA_FILTER = '*.nii';
        ZIP_IMAGE_FILES = 'No';
        icatb_runAnalysis(sesInfo, 1);"""

        result = MatlabCommand(script=script, mfile=True, prescript=[''], postscript=[''])
        r = result.run()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        prefix = self.inputs.prefix

        out_mask_image = op.abspath(prefix + 'Mask.img')
        parameter_mat_file = op.abspath(prefix + '_ica_parameter_info.mat')
        ica_mat_file = op.abspath(prefix + '_ica.mat')
        subject_mat_file = op.abspath(prefix + 'Subject.mat')
        results_log_file = op.abspath(prefix + '_results.log')

        outputs['mask_image'] = out_mask_image
        outputs['parameter_mat_file'] = parameter_mat_file
        outputs['ica_mat_file'] = ica_mat_file
        outputs['subject_mat_file'] = subject_mat_file
        outputs['results_log_file'] = results_log_file
        outputs['independent_component_images'] = glob.glob(op.abspath(op.join("components",prefix + "_component_ica_s1*")))
        outputs['independent_component_timecourse'] = glob.glob(op.abspath(op.join("components",prefix + "_timecourses_ica_s1*")))
        print outputs
        return outputs