from nipype.interfaces.matlab import MatlabCommand
from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, traits, InputMultiPath,
    OutputMultiPath, File, TraitedSpec, Directory, isdefined)
from nipype.utils.filemanip import split_filename
import os
import os.path as op
from string import Template
import nibabel as nb
import glob
import logging
import numpy as np
import random
import shutil
import scipy.io as sio
from ..helpers import analyze_to_nifti, nifti_to_analyze, switch_datatype
logging.basicConfig()
iflogger = logging.getLogger('interface')


def parse_pve_results(results_text_file):
    out_data = {}

    with open(results_text_file) as res:
        for line in res:
            if line[0:15] == "PET file name: ":
                pet_file = line[15:].strip()
                pet_file = pet_file.replace('\n', '')
            elif line[0:31] == "Labeled segmented GM file name " > 0:
                gm_file = line[31:].replace(' ', '')
                gm_file = gm_file.replace('\n', '')
            elif line[0:45] == "PET SLICE USED FOR mean WM activity MEASURE: " > 0:
                wm_slice_used = int(line[45:].replace('\n', ''))
            elif line[0:10] == "DATA OF...":
                line = line.replace(' ', '')
                line = line.replace('\n', '')
                region_names = line.split(',')[1:]
            elif line != "\r\n" and line != "\n":
                parse = [x.strip() for x in line.split(',')]
                out_data[parse[0].replace(' ', '_')] = [float(x)
                                                        for x in parse[1:]]

    out_data["wm_slice_used"] = wm_slice_used
    out_data["region_names"] = region_names
    out_data["pet_file"] = pet_file
    out_data["gm_file"] = gm_file
    return out_data


def fix_roi_values_freesurferLUT(roi_image, white_matter_file, csf_file, prob_thresh):    
    from coma.helpers import wm_labels_only, csf_labels_only, prepare_for_uint8
    _, name, _ = split_filename(roi_image)

    white_matter_default = 2
    csf_default = 3

    # Get regions labelled white and csf
    wm_label_file = op.abspath(name + "_WM.nii.gz")
    wm_only = wm_labels_only(roi_image, out_filename=wm_label_file)
    wm_label_image = nb.load(wm_only)
    wm_labels = wm_label_image.get_data()

    csf_label_file = op.abspath(name + "_CSF.nii.gz")
    csf_only = csf_labels_only(roi_image, out_filename=csf_label_file)
    csf_label_image = nb.load(csf_only)
    csf_labels = csf_label_image.get_data()

    wm_image = nb.load(white_matter_file)
    wm_data = wm_image.get_data()
    csf_image = nb.load(csf_file)
    csf_data = csf_image.get_data()

    image = nb.load(roi_image)
    data = image.get_data()

    assert(data.shape == wm_data.shape == csf_data.shape)
    data[np.where(wm_labels > 0)] = white_matter_default
    data[np.where(csf_labels > 0)] = csf_default

    # Be careful with the brackets here. & takes priority over comparisons
    data[np.where((wm_data > prob_thresh) & (data == 0))] = white_matter_default
    data[np.where((csf_data > prob_thresh) & (data == 0))] = csf_default

    wm_labels[np.where(data == white_matter_default)] = 1
    wm_labels[np.where(data == csf_default)] = 0
    wm_labels[np.where(data == 0)] = 0
    wm_labels = wm_labels.astype(np.uint8)

    new_wm_label_image = nb.Nifti1Image(
        dataobj=wm_labels, affine=wm_image.get_affine(), header=wm_image.get_header())
    new_wm_label_image.set_data_dtype(np.uint8)
    _, name, _ = split_filename(white_matter_file)
    wm_label_file = op.abspath(name + "_fixedWM.nii.gz")
    nb.save(new_wm_label_image, wm_label_file)

    csf_data[np.where(data != csf_default)] = 0
    csf_data[np.where(csf_labels > 0)] = 1

    csf_data = csf_data.astype(np.uint8)
    csf_label_image = nb.Nifti1Image(
        dataobj=csf_data, affine=csf_image.get_affine(), header=csf_image.get_header())
    csf_label_image.set_data_dtype(np.uint8)
    _, name, _ = split_filename(csf_file)
    csf_label_file = op.abspath(name + "_fixedCSF.nii.gz")
    nb.save(csf_label_image, csf_label_file)

    hdr = image.get_header()

    data_uint8, remap_dict = prepare_for_uint8(data, ignore=range(0, 3))
    data_uint8 = data_uint8.astype(np.uint8)
    data_uint8[np.where(data == csf_default)] = csf_default
    data_uint8[np.where(data == white_matter_default)] = white_matter_default

    fixed = nb.Nifti1Image(
        dataobj=data_uint8, affine=image.get_affine(), header=hdr)
    _, name, _ = split_filename(roi_image)
    fixed.set_data_dtype(np.uint8)
    fixed_roi_image = op.abspath(name + "_fixedROIs.nii.gz")
    nb.save(fixed, fixed_roi_image)
    return fixed_roi_image, wm_label_file, csf_label_file, remap_dict

def fix_roi_values_noLUT(roi_image, gm_file, white_matter_file, csf_file, prob_thresh):
    from coma.helpers import prepare_for_uint8
    _, name, _ = split_filename(roi_image)

    white_matter_default = 2
    csf_default = 3

    gm_image = nb.load(gm_file)
    gm_data = gm_image.get_data()
    wm_image = nb.load(white_matter_file)
    wm_data = wm_image.get_data()
    csf_image = nb.load(csf_file)
    csf_data = csf_image.get_data()

    image = nb.load(roi_image)
    data = image.get_data()

    assert(data.shape == gm_data.shape == wm_data.shape == csf_data.shape)

    data_uint8, remap_dict = prepare_for_uint8(data, ignore=[0])
    data_uint8 = data_uint8.astype(np.uint8)
    data_uint8[np.where(data_uint8 == csf_default)] = csf_default
    data_uint8[np.where(data_uint8 == white_matter_default)] = white_matter_default

    # Be careful with the brackets here. & takes priority over comparisons
    data_uint8[np.where((wm_data > prob_thresh) & (data_uint8 == 0))] = white_matter_default
    data_uint8[np.where((csf_data > prob_thresh) & (data_uint8 == 0))] = csf_default

    wm_data[np.where(data_uint8 == white_matter_default)] = 1
    wm_data[np.where(data_uint8 == csf_default)] = 0
    wm_data[np.where(data_uint8 == 0)] = 0
    wm_data = wm_data.astype(np.uint8)

    new_wm_label_image = nb.Nifti1Image(
        dataobj=wm_data, affine=wm_image.get_affine(), header=wm_image.get_header())
    new_wm_label_image.set_data_dtype(np.uint8)
    _, name, _ = split_filename(white_matter_file)
    wm_label_file = op.abspath(name + "_fixedWM.nii.gz")
    nb.save(new_wm_label_image, wm_label_file)

    csf_data[np.where(data_uint8 != csf_default)] = 0
    csf_data = csf_data.astype(np.uint8)
    csf_label_image = nb.Nifti1Image(
        dataobj=csf_data, affine=csf_image.get_affine(), header=csf_image.get_header())
    csf_label_image.set_data_dtype(np.uint8)
    _, name, _ = split_filename(csf_file)
    csf_label_file = op.abspath(name + "_fixedCSF.nii.gz")
    nb.save(csf_label_image, csf_label_file)

    hdr = image.get_header()

    unlabelled = np.where((gm_data > prob_thresh) & (data_uint8 == 0))[0]
    # Create extra ROI if there are extra GM regions in the GM mask
    if len(unlabelled) > 0:
        highestlabel = np.max(data_uint8)
        assert(highestlabel != 255)
        data_uint8[np.where((gm_data > prob_thresh) & (data_uint8 == 0))] = highestlabel + 1

    fixed = nb.Nifti1Image(
        dataobj=data_uint8, affine=image.get_affine(), header=hdr)
    _, name, _ = split_filename(roi_image)
    fixed.set_data_dtype(np.uint8)
    fixed_roi_image = op.abspath(name + "_fixedROIs.nii.gz")
    nb.save(fixed, fixed_roi_image)
    return fixed_roi_image, wm_label_file, csf_label_file, remap_dict


def fix_roi_values(roi_image, gm_file, white_matter_file, csf_file, use_fs_LUT=True, prob_thresh=0.7):
    '''
    Changes ROI values to prevent values equal to 1, 2,
    or 3. These are reserved for GM/WM/CSF in the PVELab
    functions.
    '''

    if use_fs_LUT:
        fixed_roi_image, wm_label_file, csf_label_file, remap_dict = fix_roi_values_freesurferLUT(roi_image, white_matter_file, csf_file, prob_thresh)
    else:
        fixed_roi_image, wm_label_file, csf_label_file, remap_dict = fix_roi_values_noLUT(roi_image, gm_file, white_matter_file, csf_file, prob_thresh)

    return fixed_roi_image, wm_label_file, csf_label_file, remap_dict


def write_config_dat(roi_file, LUT=None, remap_dict=None):
    from ..helpers import get_names
    out_file = "ROI_names.dat"
    f = open(out_file, "w")
    f.write(
        "Report ROI coding #, ROI Name and color in hexaddecimal, separated by a <Tab> hereinafter, with no space at the end\n")
    image = nb.load(roi_file)
    data = image.get_data()
    IDs = np.unique(data)
    IDs.sort()
    IDs = IDs.tolist()


    for x in xrange(4):
        if x in IDs:
            IDs.remove(x)
    if LUT is None and remap_dict is None:
        r = lambda: random.randint(0, 255)
        for idx, val in enumerate(IDs):
            # e.g. 81  R_Hippocampus       008000
            f.write("%i\tRegion%i\t%02X%02X%02X\n" % (val, idx + 1, r(), r(), r()))
        f.close()
    else:
        for x in xrange(4):
            if x in IDs:
                IDs.remove(x)
        r = lambda: random.randint(0, 255)
        for idx, val in enumerate(IDs):
            # e.g. 81  R_Hippocampus       008000
            fs_val = remap_dict[val]
            LUT_dict = get_names(LUT)
            name = LUT_dict[fs_val]
            f.write("%i\t%s\t%02X%02X%02X\n" % (val, name, r(), r(), r()))
        f.close()
    return op.abspath(out_file)


class PartialVolumeCorrectionInputSpec(BaseInterfaceInputSpec):
    pet_file = File(exists=True, mandatory=True,
                    desc='The input PET image')
    t1_file = File(exists=True, mandatory=True,
                   desc='The input T1')
    white_matter_file = File(exists=True,
                             desc='Segmented white matter')
    grey_matter_file = File(exists=True,
                            desc='Segmented grey matter')
    csf_file = File(exists=True,
                    desc='Segmented cerebrospinal fluid')
    roi_file = File(exists=True, mandatory=True, xor=['skip_atlas'],
                    desc='The input ROI image')
    skip_atlas = traits.Bool(xor=['roi_file'],
                             desc='Uses the WM/GM/CSF segmentation instead of an atlas')
    use_fs_LUT = traits.Bool(True, usedefault=True,
                             desc='Uses the Freesurfer lookup table for names in the atlas')


class PartialVolumeCorrectionOutputSpec(TraitedSpec):
    results_matlab_mat = File(
        exists=True, desc='PVELab results as MATLAB .mat file')
    results_numpy_npz = File(
        exists=True, desc='PVELab results as NumPy .npz file')
    results_text_file = File(
        exists=True, desc='PVELab results as text')
    wm_label_file = File(
        exists=True, desc='White matter binary mask from ROI labels')
    csf_label_file = File(
        exists=True, desc='Cerebrospinal fluid binary mask from ROI labels')
    alfano_alfano = File(
        exists=True, desc='alfano_alfano')
    alfano_cs = File(
        exists=True, desc='alfano_cs')
    alfano_rousset = File(
        exists=True, desc='alfano_rousset')
    mueller_gartner_alfano = File(
        exists=True, desc='mueller_gartner_alfano')
    mask = File(
        exists=True, desc='mask')
    occu_mueller_gartner = File(
        exists=True, desc='occu_mueller_gartner')
    occu_meltzer = File(
        exists=True, desc='occu_meltzer')
    meltzer = File(
        exists=True, desc='meltzer')
    mueller_gartner_rousset = File(
        exists=True, desc='mueller_gartner_rousset')
    mueller_gartner_WMroi = File(
        exists=True, desc='mueller_gartner_WMroi')
    virtual_pet_image = File(
        exists=True, desc='virtual_pet_image')
    white_matter_roi = File(
        exists=True, desc='white_matter_roi')
    rousset_mat_file = File(
        exists=True, desc='rousset_mat_file')
    point_spread_image = File(
        exists=True, desc='point_spread_image')
    out_files = OutputMultiPath(File,
                                exists=True, desc='all PVE files')


class PartialVolumeCorrection(BaseInterface):

    """
    Wraps PVELab for partial volume correction

    PVE correction includes Meltzer, M\"{u}eller-Gartner, Rousset and
    modified M\"{u}eller-Gartner (WM value estimated using Rousset method)
    approaches.

    """
    input_spec = PartialVolumeCorrectionInputSpec
    output_spec = PartialVolumeCorrectionOutputSpec

    def _run_interface(self, runtime):
        list_path = op.abspath("SubjectList.lst")
        pet_path, _ = nifti_to_analyze(self.inputs.pet_file)
        t1_path, _ = nifti_to_analyze(self.inputs.t1_file)
        f = open(list_path, 'w')
        f.write("%s;%s" % (pet_path, t1_path))
        f.close()

        orig_t1 = nb.load(self.inputs.t1_file)
        orig_affine = orig_t1.get_affine()

        gm_uint8 = switch_datatype(self.inputs.grey_matter_file)
        gm_path, _ = nifti_to_analyze(gm_uint8)
        iflogger.info("Writing to %s" % gm_path)

        fixed_roi_file, fixed_wm, fixed_csf, remap_dict = fix_roi_values(
            self.inputs.roi_file, self.inputs.grey_matter_file, 
            self.inputs.white_matter_file, self.inputs.csf_file, self.inputs.use_fs_LUT)

        rois_path, _ = nifti_to_analyze(fixed_roi_file)
        iflogger.info("Writing to %s" % rois_path)
        iflogger.info("Writing to %s" % fixed_wm)
        iflogger.info("Writing to %s" % fixed_csf)

        wm_uint8 = switch_datatype(fixed_wm)
        wm_path, _ = nifti_to_analyze(wm_uint8)
        iflogger.info("Writing to %s" % wm_path)

        csf_uint8 = switch_datatype(fixed_csf)
        csf_path, _ = nifti_to_analyze(csf_uint8)
        iflogger.info("Writing to %s" % csf_path)

        if self.inputs.use_fs_LUT:
            fs_dir = os.environ['FREESURFER_HOME']
            LUT = op.join(fs_dir, "FreeSurferColorLUT.txt")
            dat_path = write_config_dat(
                fixed_roi_file, LUT, remap_dict)
        else:
            dat_path = write_config_dat(
                fixed_roi_file)
        iflogger.info("Writing to %s" % dat_path)

        d = dict(
            list_path=list_path,
            gm_path=gm_path,
            wm_path=wm_path,
            csf_path=csf_path,
            rois_path=rois_path,
            dat_path=dat_path)
        script = Template("""       
        filelist = '$list_path';
        gm = '$gm_path';
        wm = '$wm_path';
        csf = '$csf_path';
        rois = '$rois_path';
        dat = '$dat_path';
        runbatch_nogui(filelist, gm, wm, csf, rois, dat)
        """).substitute(d)
        mlab = MatlabCommand(script=script, mfile=True,
                             prescript=[''], postscript=[''])
        result = mlab.run()

        _, foldername, _ = split_filename(self.inputs.pet_file)
        occu_MG_img = glob.glob("pve_%s/r_volume_Occu_MG.img" % foldername)[0]
        analyze_to_nifti(occu_MG_img, affine=orig_affine)
        occu_meltzer_img = glob.glob(
            "pve_%s/r_volume_Occu_Meltzer.img" % foldername)[0]
        analyze_to_nifti(occu_meltzer_img, affine=orig_affine)
        meltzer_img = glob.glob("pve_%s/r_volume_Meltzer.img" % foldername)[0]
        analyze_to_nifti(meltzer_img, affine=orig_affine)
        MG_rousset_img = glob.glob(
            "pve_%s/r_volume_MGRousset.img" % foldername)[0]
        analyze_to_nifti(MG_rousset_img, affine=orig_affine)
        MGCS_img = glob.glob("pve_%s/r_volume_MGCS.img" % foldername)[0]
        analyze_to_nifti(MGCS_img, affine=orig_affine)
        virtual_PET_img = glob.glob(
            "pve_%s/r_volume_Virtual_PET.img" % foldername)[0]
        analyze_to_nifti(virtual_PET_img, affine=orig_affine)
        centrum_semiovalue_WM_img = glob.glob(
            "pve_%s/r_volume_CSWMROI.img" % foldername)[0]
        analyze_to_nifti(centrum_semiovalue_WM_img, affine=orig_affine)
        alfano_alfano_img = glob.glob(
            "pve_%s/r_volume_AlfanoAlfano.img" % foldername)[0]
        analyze_to_nifti(alfano_alfano_img, affine=orig_affine)
        alfano_cs_img = glob.glob("pve_%s/r_volume_AlfanoCS.img" %
                                  foldername)[0]
        analyze_to_nifti(alfano_cs_img, affine=orig_affine)
        alfano_rousset_img = glob.glob(
            "pve_%s/r_volume_AlfanoRousset.img" % foldername)[0]
        analyze_to_nifti(alfano_rousset_img, affine=orig_affine)
        mg_alfano_img = glob.glob("pve_%s/r_volume_MGAlfano.img" %
                                  foldername)[0]
        analyze_to_nifti(mg_alfano_img, affine=orig_affine)
        mask_img = glob.glob("pve_%s/r_volume_Mask.img" % foldername)[0]
        analyze_to_nifti(mask_img, affine=orig_affine)
        PSF_img = glob.glob("pve_%s/r_volume_PSF.img" % foldername)[0]
        analyze_to_nifti(PSF_img)

        rousset_mat_file = glob.glob(
            "pve_%s/r_volume_Rousset.mat" % foldername)[0]
        shutil.copyfile(rousset_mat_file, op.abspath("r_volume_Rousset.mat"))

        results_text_file = glob.glob(
            "pve_%s/r_volume_pve.txt" % foldername)[0]
        shutil.copyfile(results_text_file, op.abspath("r_volume_pve.txt"))

        results_matlab_mat = op.abspath("%s_pve.mat" % foldername)
        results_numpy_npz = op.abspath("%s_pve.npz" % foldername)

        out_data = parse_pve_results(results_text_file)
        sio.savemat(results_matlab_mat, mdict=out_data)
        np.savez(results_numpy_npz, **out_data)
        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        _, foldername, _ = split_filename(self.inputs.pet_file)
        outputs['results_matlab_mat'] = op.abspath("%s_pve.mat" % foldername)
        outputs['results_numpy_npz'] = op.abspath("%s_pve.npz" % foldername)

        outputs['results_text_file'] = op.abspath("r_volume_pve.txt")
        _, name, _ = split_filename(self.inputs.white_matter_file)
        outputs['wm_label_file'] = op.abspath(name + "_fixedWM.nii.gz")
        _, name, _ = split_filename(self.inputs.csf_file)
        outputs['csf_label_file'] = op.abspath(name + "_fixedCSF.nii.gz")
        outputs['occu_mueller_gartner'] = op.abspath("r_volume_Occu_MG.nii.gz")
        outputs['occu_mueller_gartner'] = op.abspath("r_volume_Occu_MG.nii.gz")
        outputs['occu_meltzer'] = op.abspath("r_volume_Occu_Meltzer.nii.gz")
        outputs['meltzer'] = op.abspath("r_volume_Meltzer.nii.gz")
        outputs['mueller_gartner_rousset'] = op.abspath(
            "r_volume_MGRousset.nii.gz")
        outputs['mueller_gartner_WMroi'] = op.abspath("r_volume_MGCS.nii.gz")
        outputs['virtual_pet_image'] = op.abspath(
            "r_volume_Virtual_PET.nii.gz")
        outputs['white_matter_roi'] = op.abspath("r_volume_CSWMROI.nii.gz")
        outputs['rousset_mat_file'] = op.abspath("r_volume_Rousset.mat")
        outputs['point_spread_image'] = op.abspath("r_volume_PSF.nii.gz")
        outputs['mask'] = op.abspath("r_volume_Mask.nii.gz")
        outputs['alfano_alfano'] = op.abspath("r_volume_AlfanoAlfano.nii.gz")
        outputs['alfano_cs'] = op.abspath("r_volume_AlfanoCS.nii.gz")
        outputs['alfano_rousset'] = op.abspath("r_volume_AlfanoRousset.nii.gz")
        outputs['mueller_gartner_alfano'] = op.abspath(
            "r_volume_MGAlfano.nii.gz")
        outputs['out_files'] = [outputs['occu_mueller_gartner'],
                                outputs['occu_meltzer'],
                                outputs['meltzer'],
                                outputs['mueller_gartner_rousset'],
                                outputs['mueller_gartner_WMroi'],
                                outputs['virtual_pet_image'],
                                outputs['white_matter_roi'],
                                outputs['rousset_mat_file'],
                                outputs['alfano_alfano'],
                                outputs['alfano_cs'],
                                outputs['alfano_rousset'],
                                outputs['mueller_gartner_alfano'],
                                ]
        return outputs
