from nipype.interfaces.matlab import MatlabCommand
from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, traits, InputMultiPath,
    File, TraitedSpec, Directory, isdefined)
from nipype.utils.filemanip import split_filename
import os
import os.path as op
from string import Template
import nibabel as nb
import glob
import logging
import numpy as np
import random

logging.basicConfig()
iflogger = logging.getLogger('interface')

def nifti_to_analyze(nii):
    nifti = nb.load(nii)
    nif = open(nii, 'rb')
    hdr = nb.nifti1.Nifti1Header.from_fileobj(nif)
    arr_hdr = nb.analyze.AnalyzeHeader.from_header(hdr)
    arrb = hdr.raw_data_from_fileobj(nif)
    img = nb.AnalyzeImage(dataobj=arrb, affine=nifti.get_affine(), header=arr_hdr)
    _, name, _ = split_filename(nii)
    nb.analyze.save(img, op.abspath(name + '.img'))
    return op.abspath(name + '.img'), op.abspath(name + '.hdr')

def analyze_to_nifti(img, ext='.nii.gz'):
    img = nb.load(img)
    nii = nb.Nifti1Image.from_image(img)
    _, name, _ = split_filename(img)
    nb.save(nii, op.abspath(name + ext))
    return op.abspath(name + ext)

def write_config_dat(roi_file, use_fs_LUT=False):
    out_file = "ROI_names.dat"
    f = open(out_file, "w")
    f.write("Report ROI coding #, ROI Name and color in hexaddecimal, separated by a <Tab> hereinafter, with no space at the end\n")
    image = nb.load(roi_file)
    data = image.get_data()
    IDs = np.unique(data)
    IDs.sort()
    IDs = IDs.tolist()
    if 0 in IDs: IDs.remove(0)
    if 1 in IDs: IDs.remove(1)
    if 2 in IDs: IDs.remove(2)
    if 3 in IDs: IDs.remove(3)
    r = lambda: random.randint(0,255)
    for idx, val in enumerate(IDs):
        # e.g. 81  R_Hippocampus       008000
        f.write("%i\tRegion%i\t%02X%02X%02X\n" % (val, idx, r(),r(),r()))
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

        gm_path, _ = nifti_to_analyze(self.inputs.grey_matter_file)
        iflogger.info("Writing to %s" % gm_path)
        wm_path, _ = nifti_to_analyze(self.inputs.white_matter_file)
        iflogger.info("Writing to %s" % wm_path)
        csf_path, _ = nifti_to_analyze(self.inputs.csf_file)
        iflogger.info("Writing to %s" % csf_path)
        rois_path, _ = nifti_to_analyze(self.inputs.roi_file)
        iflogger.info("Writing to %s" % rois_path)

        dat_path = write_config_dat(self.inputs.roi_file, self.inputs.use_fs_LUT)
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


        occu_MG_img = glob.glob("r_volume_Occu_MG.img")[0]
        analyze_to_nifti(occu_MG_img)
        occu_meltzer_img = glob.glob("r_volume_Occu_Meltzer.img")[0]
        analyze_to_nifti(occu_meltzer_img)
        meltzer_img = glob.glob("r_volume_Meltzer.img")[0]
        analyze_to_nifti(meltzer_img)
        MG_rousset_img = glob.glob("r_volume_MGRousset.img")[0]
        analyze_to_nifti(MG_rousset_img)
        MGCS_img = glob.glob("r_volume_MGCS.img")[0]
        analyze_to_nifti(MGCS_img)
        virtual_PET_img = glob.glob("r_volume_Virtual_PET.img")[0]
        analyze_to_nifti(virtual_PET_img)
        centrum_semiovalue_WM_img = glob.glob("r_volume_CSWMROI.img")[0]
        analyze_to_nifti(centrum_semiovalue_WM_img)
        alfano_alfano_img = glob.glob("r_volume_AlfanoAlfano.img")[0]
        analyze_to_nifti(alfano_alfano_img)
        alfano_cs_img = glob.glob("r_volume_AlfanoCS.img")[0]
        analyze_to_nifti(alfano_cs_img)
        alfano_rousset_img = glob.glob("r_volume_AlfanoRousset.img")[0]
        analyze_to_nifti(alfano_rousset_img)
        mg_alfano_img = glob.glob("r_volume_MGAlfano.img")[0]
        analyze_to_nifti(mg_alfano_img)
        mask_img = glob.glob("r_volume_Mask.img")[0]
        analyze_to_nifti(mask_img)
        PSF_img = glob.glob("r_volume_PSF.img")[0]
        analyze_to_nifti(PSF_img)
        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['occu_mueller_gartner'] = op.abspath("r_volume_Occu_MG.nii")
        outputs['occu_meltzer'] = op.abspath("r_volume_Occu_Meltzer.nii")
        outputs['meltzer'] = op.abspath("r_volume_Meltzer.nii")
        outputs['mueller_gartner_rousset'] = op.abspath("r_volume_MGRousset.nii")
        outputs['mueller_gartner_WMroi'] = op.abspath("r_volume_MGCS.nii")
        outputs['virtual_pet_image'] = op.abspath("r_volume_Virtual_PET.nii")
        outputs['white_matter_roi'] = op.abspath("r_volume_CSWMROI.nii")
        outputs['rousset_mat_file'] = op.abspath("r_volume_Rousset.mat")
        outputs['point_spread_image'] = op.abspath("r_volume_PSF.nii")
        outputs['mask'] = op.abspath("r_volume_Mask.nii")
        outputs['alfano_alfano'] = op.abspath("r_volume_AlfanoAlfano.nii")
        outputs['alfano_cs'] = op.abspath("r_volume_AlfanoCS.nii")
        outputs['alfano_rousset'] = op.abspath("r_volume_AlfanoRousset.nii")
        outputs['mueller_gartner_alfano'] = op.abspath("r_volume_MGAlfano.nii")
        return outputs