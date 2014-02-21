import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.mrtrix as mrtrix
import nipype.interfaces.cmtk as cmtk
from nipype.interfaces.utility import Function
from nipype.workflows.misc.utils import select_aparc

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from coma.interfaces import RegionalValues


def extract_PreCoTh(in_file):
    from nipype.utils.filemanip import split_filename
    import nibabel as nb
    import numpy as np
    import os.path as op
    in_image = nb.load(in_file)
    in_header = in_image.get_header()
    in_data = in_image.get_data()

    #Left, Right
    # Thalamus are 76 and 35
    # Precuneus are 61 and 20
    # Cortex 3 and 42

    MAPPING = [
        [42, 2012], [42, 2019], [42, 2032], [42, 2014], [42, 2020], [42, 2018],
        [42,2027], [42, 2028], [42, 2003], [42, 2024], [42, 2017], [42, 2026],
        [42, 2002], [42, 2023], [42, 2010], [42, 2022], [42, 2031], [42,2029],
        [42, 2008], [42, 2005], [42, 2021], [42, 2011],
        [42, 2013], [42, 2007], [42, 2016], [42, 2006], [42, 2033], [42,2009],
        [42, 2015], [42, 2001], [42, 2030], [42, 2034], [42, 2035],
        [3, 1012], [3, 1019], [3, 1032], [3, 1014], [3, 1020], [3, 1018],
        [3, 1027], [3, 1028], [3, 1003], [3, 1024], [3, 1017], [3,1026],
        [3, 1002], [3, 1023], [3, 1010], [3, 1022], [3, 1031],
        [3, 1029], [3, 1008],  [3, 1005], [3, 1021], [3, 1011], [3,1013],
        [3, 1007], [3, 1016], [3, 1006], [3, 1033],
        [3, 1009], [3, 1015], [3, 1001], [3, 1030], [3, 1034], [3, 1035],
        [61, 1025], [76, 10], [20, 2025], [35, 49]]

    niiGM = np.zeros(in_data.shape, dtype=np.uint)
    for ma in MAPPING:
        # print ma
        niiGM[in_data == ma[1]] = ma[0]

    _, name, _ = split_filename(in_file)
    out_file = op.abspath(name) + "_pre_co_th.nii.gz"
    out_image = nb.Nifti1Image(
        data=niiGM, header=in_header, affine=in_image.get_affine())
    nb.save(out_image, out_file)
    return out_file


def create_precoth_pipeline(name="precoth", tractography_type='probabilistic'):
    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=["subjects_dir",
                                                 "subject_id",
                                                 "dwi",
                                                 "fdgpet",
                                                 "bvecs",
                                                 "bvals"]),
        name="inputnode")

    bet = pe.Node(interface=fsl.BET(), name="bet")
    bet.inputs.mask = True

    fsl2mrtrix = pe.Node(interface=mrtrix.FSL2MRTrix(), name='fsl2mrtrix')
    fsl2mrtrix.inputs.invert_y = True

    dwi2tensor = pe.Node(interface=mrtrix.DWI2Tensor(), name='dwi2tensor')

    tensor2fa = pe.Node(interface=mrtrix.Tensor2FractionalAnisotropy(),
                        name='tensor2fa')

    tensor2md = pe.Node(interface=mrtrix.Tensor2ApparentDiffusion(),
                        name='tensor2md')

    erode_mask_firstpass = pe.Node(interface=mrtrix.Erode(),
                                   name='erode_mask_firstpass')
    erode_mask_secondpass = pe.Node(interface=mrtrix.Erode(),
                                    name='erode_mask_secondpass')

    threshold_b0 = pe.Node(interface=mrtrix.Threshold(), name='threshold_b0')

    threshold_FA = pe.Node(interface=mrtrix.Threshold(), name='threshold_FA')
    threshold_FA.inputs.absolute_threshold_value = 0.7

    threshold_wmmask = pe.Node(interface=mrtrix.Threshold(),
                               name='threshold_wmmask')
    threshold_wmmask.inputs.absolute_threshold_value = 0.4

    MRmultiply = pe.Node(interface=mrtrix.MRMultiply(), name='MRmultiply')
    MRmult_merge = pe.Node(interface=util.Merge(2), name='MRmultiply_merge')

    median3d = pe.Node(interface=mrtrix.MedianFilter3D(), name='median3D')

    fdgpet_regions = pe.Node(interface=RegionalValues(), name='fdgpet_regions')

    MRconvert = pe.Node(interface=mrtrix.MRConvert(), name='MRconvert')
    MRconvert.inputs.extract_at_axis = 3
    MRconvert.inputs.extract_at_coordinate = [0]

    csdeconv = pe.Node(interface=mrtrix.ConstrainedSphericalDeconvolution(),
                       name='csdeconv')

    gen_WM_mask = pe.Node(interface=mrtrix.GenerateWhiteMatterMask(),
                          name='gen_WM_mask')

    estimateresponse = pe.Node(interface=mrtrix.EstimateResponseForSH(),
                               name='estimateresponse')

    if tractography_type == 'probabilistic':
        CSDstreamtrack = pe.Node(
            interface=mrtrix.ProbabilisticSphericallyDeconvolutedStreamlineTrack(
            ),
            name='CSDstreamtrack')
    else:
        CSDstreamtrack = pe.Node(
            interface=mrtrix.SphericallyDeconvolutedStreamlineTrack(),
            name='CSDstreamtrack')

    CSDstreamtrack.inputs.desired_number_of_tracks = 300000

    fa_to_nii = pe.Node(interface=mrtrix.MRConvert(), name='fa_to_nii')
    fa_to_nii.inputs.extension = 'nii'
    md_to_nii = fa_to_nii.clone('md_to_nii')

    tck2trk = pe.Node(interface=mrtrix.MRTrix2TrackVis(), name='tck2trk')

    extract_PreCoTh_interface = Function(input_names=["in_file"],
                                         output_names=["out_file"],
                                         function=extract_PreCoTh)

    thalamus2precuneus2cortex_ROIs = pe.Node(
        interface=extract_PreCoTh_interface, name='thalamus2precuneus2cortex_ROIs')

    thalamus2precuneus2cortex = pe.Node(
        interface=cmtk.CreateMatrix(), name="thalamus2precuneus2cortex")
    thalamus2precuneus2cortex.inputs.count_region_intersections = False

    FreeSurferSource = pe.Node(
        interface=nio.FreeSurferSource(), name='fssource')
    mri_convert_Brain = pe.Node(
        interface=fs.MRIConvert(), name='mri_convert_Brain')
    mri_convert_Brain.inputs.out_type = 'niigz'

    mri_convert_ROIs = mri_convert_Brain.clone("mri_convert_ROIs")

    reslice_fdgpet = mri_convert_Brain.clone("reslice_fdgpet")

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    workflow.connect(
        [(inputnode, FreeSurferSource, [("subjects_dir", "subjects_dir")])])
    workflow.connect(
        [(inputnode, FreeSurferSource, [("subject_id", "subject_id")])])
    workflow.connect(
        [(FreeSurferSource, mri_convert_ROIs, [(('aparc_aseg', select_aparc), 'in_file')])])

    workflow.connect(
        [(FreeSurferSource, mri_convert_Brain, [('brain', 'in_file')])])
    workflow.connect([(inputnode, fsl2mrtrix, [("bvecs", "bvec_file"),
                                               ("bvals", "bval_file")])])
    workflow.connect([(inputnode, dwi2tensor, [("dwi", "in_file")])])
    workflow.connect(
        [(fsl2mrtrix, dwi2tensor, [("encoding_file", "encoding_file")])])
    workflow.connect([(dwi2tensor, tensor2fa, [("tensor", "in_file")])])
    workflow.connect([(tensor2fa, fa_to_nii, [("FA", "in_file")])])

    workflow.connect([(dwi2tensor, tensor2md, [("tensor", "in_file")])])
    workflow.connect([(tensor2md, md_to_nii, [("ADC", "in_file")])])

    workflow.connect([(inputnode, reslice_fdgpet, [("fdgpet", "in_file")])])
    workflow.connect(
        [(mri_convert_ROIs, reslice_fdgpet, [("out_file", "reslice_like")])])
    workflow.connect(
        [(reslice_fdgpet, fdgpet_regions, [("out_file", "in_files")])])
    workflow.connect(
        [(thalamus2precuneus2cortex_ROIs, fdgpet_regions, [("out_file", "segmentation_file")])])

    workflow.connect([(inputnode, MRconvert, [("dwi", "in_file")])])
    workflow.connect([(MRconvert, threshold_b0, [("converted", "in_file")])])
    workflow.connect([(threshold_b0, median3d, [("out_file", "in_file")])])
    workflow.connect(
        [(median3d, erode_mask_firstpass, [("out_file", "in_file")])])
    workflow.connect(
        [(erode_mask_firstpass, erode_mask_secondpass, [("out_file", "in_file")])])

    workflow.connect([(tensor2fa, MRmult_merge, [("FA", "in1")])])
    workflow.connect(
        [(erode_mask_secondpass, MRmult_merge, [("out_file", "in2")])])
    workflow.connect([(MRmult_merge, MRmultiply, [("out", "in_files")])])
    workflow.connect([(MRmultiply, threshold_FA, [("out_file", "in_file")])])
    workflow.connect(
        [(threshold_FA, estimateresponse, [("out_file", "mask_image")])])

    workflow.connect([(inputnode, bet, [("dwi", "in_file")])])
    workflow.connect([(inputnode, gen_WM_mask, [("dwi", "in_file")])])
    workflow.connect([(bet, gen_WM_mask, [("mask_file", "binary_mask")])])
    workflow.connect(
        [(fsl2mrtrix, gen_WM_mask, [("encoding_file", "encoding_file")])])

    workflow.connect([(inputnode, estimateresponse, [("dwi", "in_file")])])
    workflow.connect(
        [(fsl2mrtrix, estimateresponse, [("encoding_file", "encoding_file")])])

    workflow.connect([(inputnode, csdeconv, [("dwi", "in_file")])])
    workflow.connect(
        [(gen_WM_mask, csdeconv, [("WMprobabilitymap", "mask_image")])])
    workflow.connect(
        [(estimateresponse, csdeconv, [("response", "response_file")])])
    workflow.connect(
        [(fsl2mrtrix, csdeconv, [("encoding_file", "encoding_file")])])

    workflow.connect(
        [(gen_WM_mask, threshold_wmmask, [("WMprobabilitymap", "in_file")])])
    workflow.connect(
        [(threshold_wmmask, CSDstreamtrack, [("out_file", "seed_file")])])
    workflow.connect(
        [(csdeconv, CSDstreamtrack, [("spherical_harmonics_image", "in_file")])])

    workflow.connect([(CSDstreamtrack, tck2trk, [("tracks", "in_file")])])
    workflow.connect([(inputnode, tck2trk, [("dwi", "image_file")])])
    # Needs reg to T1

    workflow.connect(
        [(tck2trk, thalamus2precuneus2cortex, [("out_file", "tract_file")])])
    workflow.connect(
        [(inputnode, thalamus2precuneus2cortex, [("subject_id", "out_matrix_file")])])
    workflow.connect(
        [(inputnode, thalamus2precuneus2cortex, [("subject_id", "out_matrix_mat_file")])])

    workflow.connect(
        [(mri_convert_ROIs, thalamus2precuneus2cortex_ROIs, [("out_file", "in_file")])])
    workflow.connect(
        [(thalamus2precuneus2cortex_ROIs, thalamus2precuneus2cortex, [("out_file", "roi_file")])])

    output_fields = ["fa", "md", "csdeconv", "tracts_tck"]

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=output_fields),
        name="outputnode")

    workflow.connect(
        [(CSDstreamtrack, outputnode, [("tracks", "tracts_tck")]),
         (csdeconv, outputnode,
          [("spherical_harmonics_image", "csdeconv")]),
         (tensor2fa, outputnode, [("out_files", "fa")]),
         (tensor2md, outputnode, [("out_files", "md")])])

    return workflow
