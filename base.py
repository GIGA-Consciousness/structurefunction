from nipype.interfaces.matlab import MatlabCommand
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec, traits, InputMultiPath,
                                    File, TraitedSpec, Directory, isdefined, OutputMultiPath)
from nipype.utils.filemanip import split_filename, list_to_filename
import os, os.path as op
import re
from string import Template
import numpy as np
import nibabel as nb
import networkx as nx
import scipy.io as sio
import shutil
import logging

logging.basicConfig()
iflogger = logging.getLogger('interface')

try: 
	os.environ['COMA_REST_LIB_ROOT']
except KeyError:
	iflogger.error('COMA_REST_LIB_ROOT environment variable not set.')

class CreateDenoisedImageInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True,
    desc='The input ICA maps as separate images')
    in_file4d = File(exists=True, mandatory=True, xor=['in_files'],
    desc='The input ICA maps as a single four-dimensional file')
    time_course_image = File(exists=True, mandatory=True, desc='The fMRI time course as an image')
    ica_mask_image = File(exists=True, mandatory=True, desc='The ICA mask image')
    coma_rest_lib_path = Directory(exists=True, mandatory=True, desc='The ICA mask image')
    repetition_time = traits.Float(mandatory=True, desc='The repetition time (TR) in seconds')
    out_neuronal_image = File('neuronal.nii', usedefault=True, desc='Reconstructed "denoised" image from neuronal components')
    out_non_neuronal_image = File('non_neuronal.nii', usedefault=True, desc='Reconstructed "noise" image from non-neuronal components')

class CreateDenoisedImageOutputSpec(TraitedSpec):
    neuronal_image = File(exists=True, desc='Reconstructed "denoised" image from neuronal components')
    non_neuronal_image = File(exists=True, desc='Reconstructed "noise" image from non-neuronal components')

class CreateDenoisedImage(BaseInterface):
    """
    Wraps a MATLAB/Java program for constructing a denoised fMRI image from a set of independent component analysis maps 
    that were generated from resting-state fMRI and have been classified into neuronal and non-neuronal components.

    """
    input_spec = CreateDenoisedImageInputSpec
    output_spec = CreateDenoisedImageOutputSpec

    def _run_interface(self, runtime):
        data_dir = op.abspath('./denoise/components')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        in_files = self.inputs.in_files
        if len(self.inputs.in_files) > 1:
            print 'Multiple ({n}) input images detected! Copying to {d}...'.format(n=len(self.inputs.in_files), d=data_dir)
            for in_file in self.inputs.in_files:
                path, name, ext = split_filename(in_file)
                shutil.copyfile(in_file, op.join(data_dir, name) + ext)
                if ext == '.img':
                    shutil.copyfile(op.join(path, name) + '.hdr', op.join(data_dir, name) + '.hdr')
                elif ext == '.hdr':
                    shutil.copyfile(op.join(path, name) + '.img', op.join(data_dir, name) + '.img')
            print 'Copied!'
            in_files = self.inputs.in_files
        elif isdefined(self.inputs.in_file4d):
            print 'Single four-dimensional image selected. Splitting and copying to {d}'.format(d=data_dir)
            in_files = nb.four_to_three(self.inputs.in_file4d)
            for in_file in in_files:
                path, name, ext = split_filename(in_file)
                shutil.copyfile(in_file, op.join(data_dir, name) + ext)
            print 'Copied!'
        else:
            print 'Single functional image provided. Ending...'
            in_files = self.inputs.in_files

        nComponents = len(in_files)
        path, name, ext = split_filename(self.inputs.time_course_image)
        shutil.copyfile(self.inputs.time_course_image, op.join(data_dir, name) + ext)
        if ext == '.img':
            shutil.copyfile(op.join(path, name) + '.hdr', op.join(data_dir, name) + '.hdr')
        elif ext == '.hdr':
            shutil.copyfile(op.join(path, name) + '.img', op.join(data_dir, name) + '.img')

        data_dir = op.abspath('./denoise')
        path, name, ext = split_filename(self.inputs.ica_mask_image)
        shutil.copyfile(self.inputs.ica_mask_image, op.join(data_dir, name) + ext)
        if ext == '.img':
            shutil.copyfile(op.join(path, name) + '.hdr', op.join(data_dir, name) + '.hdr')
        elif ext == '.hdr':
            shutil.copyfile(op.join(path, name) + '.img', op.join(data_dir, name) + '.img')
        mask_file = op.join(data_dir, name)
        repetition_time = self.inputs.repetition_time
        neuronal_image = op.abspath(self.inputs.out_neuronal_image)
        non_neuronal_image = op.abspath(self.inputs.out_non_neuronal_image)
        coma_rest_lib_path = op.abspath(self.inputs.coma_rest_lib_path)
        d = dict(data_dir=data_dir, mask_name=mask_file, nComponents=nComponents, Tr=repetition_time, nameNeuronal=neuronal_image, nameNonNeuronal=non_neuronal_image, coma_rest_lib_path=coma_rest_lib_path)
        script = Template("""
        restlib_path = '$coma_rest_lib_path';
        setup_restlib_paths(restlib_path)
        dataDir = '$data_dir';
        maskName = '$mask_name';
        nCompo = $nComponents;
        Tr = $Tr;
        nameNeuronalData = '$nameNeuronal';
        nameNonNeuronalData = '$nameNonNeuronal';
        denoiseImage(dataDir,maskName,nCompo,Tr,nameNeuronalData,nameNonNeuronalData, restlib_path);
        """).substitute(d)
        result = MatlabCommand(script=script, mfile=True, prescript=[''], postscript=[''])
        r = result.run()
        print 'Neuronal component image saved as {n}'.format(n=neuronal_image)
        print 'Non-neuronal component image saved as {n}'.format(n=non_neuronal_image)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['neuronal_image'] = op.abspath(self.inputs.out_neuronal_image)
        outputs['non_neuronal_image'] = op.abspath(self.inputs.out_non_neuronal_image)
        return outputs

class MatchingClassificationInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True,
    desc='The input ICA maps as separate images')
    in_file4d = File(exists=True, mandatory=True, xor=['in_files'],
    desc='The input ICA maps as a single four-dimensional file')
    ica_mask_image = File(exists=True, mandatory=True, desc='The ICA mask image')
    time_course_image = File(exists=True, mandatory=True, desc='The fMRI time course as an image')
    coma_rest_lib_path = Directory(exists=True, mandatory=True, desc='The ICA mask image')
    repetition_time = traits.Float(mandatory=True, desc='The repetition time (TR) in seconds')
    out_stats_file = File('stats.mat', usedefault=True, desc='Reconstructed "denoised" image from neuronal components')
    
class MatchingClassificationOutputSpec(TraitedSpec):
    stats_file = File(exists=True, desc='Statistics file')

class MatchingClassification(BaseInterface):
    """
    Wraps a MATLAB/Java program for classifying independent component analysis maps from resting-state fMRI into neuronal and non-neuronal components.

    """
    input_spec = MatchingClassificationInputSpec
    output_spec = MatchingClassificationOutputSpec

    def _run_interface(self, runtime):
        path, name, ext = split_filename(self.inputs.time_course_image)
        data_dir = op.abspath('./matching')
        copy_to = op.join(data_dir, 'components')
        if not op.exists(copy_to):
            os.makedirs(copy_to)
        copy_to = op.join(copy_to,name)
        shutil.copyfile(self.inputs.time_course_image, copy_to + ext)
        if ext == '.img':
            shutil.copyfile(op.join(path, name) + '.hdr', copy_to + '.hdr')
        elif ext == '.hdr':
            shutil.copyfile(op.join(path, name) + '.img', copy_to + '.img')
        time_course_file = copy_to + '.img'

        data_dir = op.abspath('./matching/components')
        in_files = self.inputs.in_files
        if len(self.inputs.in_files) > 1:
            print 'Multiple ({n}) input images detected! Copying to {d}...'.format(n=len(self.inputs.in_files), d=data_dir)
            for in_file in self.inputs.in_files:
                path, name, ext = split_filename(in_file)
                shutil.copyfile(in_file, op.join(data_dir, name) + ext)
                if ext == '.img':
                    shutil.copyfile(op.join(path, name) + '.hdr', op.join(data_dir, name) + '.hdr')
                elif ext == '.hdr':
                    shutil.copyfile(op.join(path, name) + '.img', op.join(data_dir, name) + '.img')
            print 'Copied!'
        elif isdefined(self.inputs.in_file4d):
            print 'Single four-dimensional image selected. Splitting and copying to {d}'.format(d=data_dir)
            in_files = nb.four_to_three(self.inputs.in_file4d)
            for in_file in in_files:
                path, name, ext = split_filename(in_file)
                shutil.copyfile(in_file, op.join(data_dir, name) + ext)
            print 'Copied!'
        else:
            print 'Single functional image provided. Ending...'
            in_files = self.inputs.in_files
        
        nComponents = len(in_files)
        repetition_time = self.inputs.repetition_time
        coma_rest_lib_path = op.abspath(self.inputs.coma_rest_lib_path)

        data_dir = op.abspath('./matching')
        if not op.exists(data_dir):
            os.makedirs(data_dir)
            
        path, name, ext = split_filename(self.inputs.ica_mask_image)
        copy_to = op.join(data_dir, 'components')
        if not op.exists(copy_to):
            os.makedirs(copy_to)
        copy_to = op.join(copy_to,name)
        shutil.copyfile(self.inputs.ica_mask_image, copy_to + ext)
        if ext == '.img':
            shutil.copyfile(op.join(path, name) + '.hdr', copy_to + '.hdr')
        elif ext == '.hdr':
            shutil.copyfile(op.join(path, name) + '.img', copy_to + '.img')
        mask_file = op.abspath(self.inputs.ica_mask_image)
        out_stats_file = op.abspath(self.inputs.out_stats_file)
        d = dict(out_stats_file=out_stats_file, data_dir=data_dir, mask_name=mask_file, nComponents=nComponents, Tr=repetition_time, coma_rest_lib_path=coma_rest_lib_path)
        script = Template("""
        restlib_path = '$coma_rest_lib_path';
        setup_restlib_paths(restlib_path)
        namesTemplate = {'rAuditory_corr','rCerebellum_corr','rDMN_corr','rECN_L_corr','rECN_R_corr','rSalience_corr','rSensorimotor_corr','rVisual_lateral_corr','rVisual_medial_corr','rVisual_occipital_corr'};
        indexNeuronal = 1:$nComponents;
        nCompo = $nComponents;
        out_stats_file = '$out_stats_file'; 
        Tr = $Tr;
        data_dir = '$data_dir'
        mask_name = '$mask_name'
        [dataAssig maxGoF] = selectionMatchClassification(data_dir, mask_name, namesTemplate,indexNeuronal,nCompo,Tr,restlib_path)
                            
        for i=1:size(dataAssig,1)
            str{i} = sprintf('Template %d: %s to component %d with GoF %f is neuronal %d prob=%f',dataAssig(i,1),namesTemplate{i},dataAssig(i,2),dataAssig(i,3),dataAssig(i,4),dataAssig(i,5));
            disp(str{i});
        end
        maxGoF
        templates = dataAssig(:,1)
        components = dataAssig(:,2)
        gofs = dataAssig(:,3)
        neuronal_bool = dataAssig(:,4)
        neuronal_prob = dataAssig(:,5)
        save '$out_stats_file'
        """).substitute(d)
        print 'Saving stats file as {s}'.format(s=out_stats_file)
        result = MatlabCommand(script=script, mfile=True, prescript=[''], postscript=[''])
        r = result.run()
        #matlab_file = sio.loadmat(out_stats_file)
        #data_assig = matlab_file['dataAssig']
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['stats_file'] = op.abspath(self.inputs.out_stats_file)
        return outputs

class ComputeFingerprintInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
    desc='The input ICA maps as separate images')
    ica_mask_image = File(exists=True, mandatory=True, desc='The ICA mask image')
    time_course_image = File(exists=True, mandatory=True, desc='The fMRI time course as an image')
    coma_rest_lib_path = Directory(exists=True, mandatory=True, desc='The ICA mask image')
    repetition_time = traits.Float(mandatory=True, desc='The repetition time (TR) in seconds')
    subject_id = traits.Str(desc='Subject ID')
    component_index = traits.Int(desc='Index of the independent component to use from the t-value threshold file.')
    out_stats_file = File(desc='Reconstructed "denoised" image from neuronal components')

class ComputeFingerprintOutputSpec(TraitedSpec):
    stats_file = File(exists=True, desc='Neuronal component images')

class ComputeFingerprint(BaseInterface):
    """
    Computes a fingerprint for the ICA component

    """
    input_spec = ComputeFingerprintInputSpec
    output_spec = ComputeFingerprintOutputSpec

    def _run_interface(self, runtime):
        path, name, ext = split_filename(self.inputs.time_course_image)
        data_dir = op.abspath('./matching')
        copy_to = op.join(data_dir, 'components')
        if not op.exists(copy_to):
            os.makedirs(copy_to)
        copy_to = op.join(copy_to,name)
        shutil.copyfile(self.inputs.time_course_image, copy_to + ext)
        if ext == '.img':
            shutil.copyfile(op.join(path, name) + '.hdr', copy_to + '.hdr')
        elif ext == '.hdr':
            shutil.copyfile(op.join(path, name) + '.img', copy_to + '.img')
        time_course_file = copy_to + '.img'

        path, name, ext = split_filename(self.inputs.ica_mask_image)
        shutil.copyfile(self.inputs.ica_mask_image, op.join(data_dir, name) + ext)
        if ext == '.img':
            shutil.copyfile(op.join(path, name) + '.hdr', op.join(data_dir, name) + '.hdr')
        elif ext == '.hdr':
            shutil.copyfile(op.join(path, name) + '.img', op.join(data_dir, name) + '.img')
        mask_file = op.abspath(self.inputs.ica_mask_image)
        repetition_time = self.inputs.repetition_time
        component_file = op.abspath(self.inputs.in_file)
        coma_rest_lib_path = op.abspath(self.inputs.coma_rest_lib_path)
        component_index = self.inputs.component_index
        if isdefined(self.inputs.out_stats_file):
            path, name, ext = split_filename(self.inputs.out_stats_file)
            if not ext == '.mat':
                ext = '.mat'
            out_stats_file = op.abspath(name+ext)            
        else:
            if isdefined(self.inputs.subject_id):
                out_stats_file = op.abspath(self.inputs.subject_id + '_IC_' + str(self.inputs.component_index) + '.mat')
            else:
                out_stats_file = op.abspath('IC_' + str(self.inputs.component_index) + '.mat')        
        
        d = dict(component_file=component_file, IC=component_index, time_course_file=time_course_file, mask_name=mask_file, Tr=repetition_time, coma_rest_lib_path=coma_rest_lib_path, out_stats_file=out_stats_file)
        script = Template("""
        restlib_path = '$coma_rest_lib_path';
        setup_restlib_paths(restlib_path);
        Tr = $Tr;
        out_stats_file = '$out_stats_file';
        component_file = '$component_file';
        maskName = '$mask_name';
        maskData = load_nii(maskName);        
        dataCompSpatial = load_nii(component_file)
        time_course_file = '$time_course_file'
        timeData = load_nii(time_course_file)
        IC = $IC
        [feature dataZ temporalData] = computeFingerprintSpaceTime(dataCompSpatial.img,timeData.img(:,IC),maskData.img,Tr);
        save '$out_stats_file'
        """).substitute(d)
        result = MatlabCommand(script=script, mfile=True, prescript=[''], postscript=[''])
        r = result.run()
        print 'Saving stats file as {s}'.format(s=out_stats_file)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.out_stats_file):
            path, name, ext = split_filename(self.inputs.out_stats_file)
            if not ext == '.mat':
                ext = '.mat'
            out_stats_file = op.abspath(name+ext)            
        else:
            if isdefined(self.inputs.subject_id):
                out_stats_file = op.abspath(self.inputs.subject_id + '_IC_' + str(self.inputs.component_index) + '.mat')
            else:
                out_stats_file = op.abspath('IC_' + str(self.inputs.component_index) + '.mat')
        outputs['stats_file'] = out_stats_file
        return outputs

