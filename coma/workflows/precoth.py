import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.mrtrix as mrtrix
import nipype.interfaces.cmtk as cmtk
from nipype.interfaces.utility import Function
from nipype.workflows.misc.utils import select_aparc
import nipype.interfaces.ants as ants

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from coma.interfaces import RegionalValues, nonlinfit_fn


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

    # MAPPING = [
    #     [42, 2012], [42, 2019], [42, 2032], [42, 2014], [42, 2020], [42, 2018],
    #     [42,2027], [42, 2028], [42, 2003], [42, 2024], [42, 2017], [42, 2026],
    #     [42, 2002], [42, 2023], [42, 2010], [42, 2022], [42, 2031], [42,2029],
    #     [42, 2008], [42, 2005], [42, 2021], [42, 2011],
    #     [42, 2013], [42, 2007], [42, 2016], [42, 2006], [42, 2033], [42,2009],
    #     [42, 2015], [42, 2001], [42, 2030], [42, 2034], [42, 2035],
    #     [3, 1012], [3, 1019], [3, 1032], [3, 1014], [3, 1020], [3, 1018],
    #     [3, 1027], [3, 1028], [3, 1003], [3, 1024], [3, 1017], [3,1026],
    #     [3, 1002], [3, 1023], [3, 1010], [3, 1022], [3, 1031],
    #     [3, 1029], [3, 1008],  [3, 1005], [3, 1021], [3, 1011], [3,1013],
    #     [3, 1007], [3, 1016], [3, 1006], [3, 1033],
    #     [3, 1009], [3, 1015], [3, 1001], [3, 1030], [3, 1034], [3, 1035],
    #     [61, 1025], [76, 10], [20, 2025], [35, 49]]

    # Left, Right -> Now
    # Thalamus are 71 and 35
    # Thal L-1 R-2
    # Precuneus are 61 and 20
    # Prec L-5 R-6
    # Cortex 3 and 42
    # Cortex 3 and 4

    MAPPING = [
        [4, 2012], [4, 2019], [4, 2032], [4, 2014], [4, 2020], [4, 2018],
        [4, 2027], [4, 2028], [4, 2003], [4, 2024], [4, 2017], [4, 2026],
        [4, 2002], [4, 2023], [4, 2010], [4, 2022], [4, 2031], [4, 2029],
        [4, 2008], [4, 2005], [4, 2021], [4, 2011],
        [4, 2013], [4, 2007], [4, 2016], [4, 2006], [4, 2033], [4, 2009],
        [4, 2015], [4, 2001], [4, 2030], [4, 2034], [4, 2035],

        [3, 1012], [3, 1019], [3, 1032], [3, 1014], [3, 1020], [3, 1018],
        [3, 1027], [3, 1028], [3, 1003], [3, 1024], [3, 1017], [3, 1026],
        [3, 1002], [3, 1023], [3, 1010], [3, 1022], [3, 1031],
        [3, 1029], [3, 1008], [3, 1005], [3, 1021], [3, 1011], [3,1013],
        [3, 1007], [3, 1016], [3, 1006], [3, 1033],
        [3, 1009], [3, 1015], [3, 1001], [3, 1030], [3, 1034], [3, 1035],

        [5, 1025], [6, 2025], [1, 10], [2, 49]]

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

    nonlinfit_interface = util.Function(input_names=["dwi", "bvecs", "bvals", "base_name"],
    output_names=["tensor", "FA", "MD", "evecs", "evals", "rgb_fa", "norm", "mode", "binary_mask", "b0_masked"], function=nonlinfit_fn)

    nonlinfit_node = pe.Node(interface=nonlinfit_interface, name="nonlinfit_node")


    coregister = pe.Node(interface=fsl.FLIRT(dof=12), name = 'coregister')
    coregister.inputs.cost = ('normmi')

    invertxfm = pe.Node(interface=fsl.ConvertXFM(), name = 'invertxfm')
    invertxfm.inputs.invert_xfm = True

    WM_to_FA = pe.Node(interface=fsl.ApplyXfm(), name = 'WM_to_FA')


    # FA_to_T1 = pe.Node(interface=ants.Registration(), name="FA_to_T1")
    # FA_to_T1.inputs.output_transform_prefix = "FA_to_T1_"

    # FA_to_T1.inputs.transforms = ['Affine']#, 'SyN']
    # FA_to_T1.inputs.transform_parameters = [(2.0,)]#, (0.25, 3.0, 0.0)]
    # FA_to_T1.inputs.number_of_iterations = [[1500, 200]]#, [100, 50, 30]]
    # FA_to_T1.inputs.dimension = 3
    # FA_to_T1.inputs.write_composite_transform = True
    # FA_to_T1.inputs.collapse_output_transforms = False
    # FA_to_T1.inputs.metric = ['Mattes']#*2
    # FA_to_T1.inputs.metric_weight = [1]#*2 # Default (value ignored currently by ANTs)
    # FA_to_T1.inputs.radius_or_number_of_bins = [32]#*2
    # FA_to_T1.inputs.sampling_strategy = ['Random']#, None]
    # FA_to_T1.inputs.sampling_percentage = [0.05]#, None]
    # FA_to_T1.inputs.convergence_threshold = [1.e-8]#, 1.e-9]
    # FA_to_T1.inputs.convergence_window_size = [20]#*2
    # FA_to_T1.inputs.smoothing_sigmas = [[1,0]]#, [2,1,0]]
    # FA_to_T1.inputs.sigma_units = ['vox']# * 2
    # FA_to_T1.inputs.shrink_factors = [[2,1]]#, [3,2,1]]
    # FA_to_T1.inputs.use_estimate_learning_rate_once = [True]#, True]
    # FA_to_T1.inputs.use_histogram_matching = [True]#, True] # This is the default
    # FA_to_T1.inputs.output_warped_image = 'FA_T1space.nii.gz'
    # #FA_to_T1.inputs.num_threads = 2

    # # FA_to_T1.inputs.transforms = ['Affine', 'SyN']
    # # FA_to_T1.inputs.transform_parameters = [(2.0,), (0.25, 3.0, 0.0)]
    # # FA_to_T1.inputs.number_of_iterations = [[1500, 200], [100, 50, 30]]
    # # FA_to_T1.inputs.dimension = 3
    # # FA_to_T1.inputs.write_composite_transform = True
    # # FA_to_T1.inputs.collapse_output_transforms = False
    # # FA_to_T1.inputs.metric = ['Mattes']*2
    # # FA_to_T1.inputs.metric_weight = [1]*2 # Default (value ignored currently by ANTs)
    # # FA_to_T1.inputs.radius_or_number_of_bins = [32]*2
    # # FA_to_T1.inputs.sampling_strategy = ['Random', None]
    # # FA_to_T1.inputs.sampling_percentage = [0.05, None]
    # # FA_to_T1.inputs.convergence_threshold = [1.e-8, 1.e-9]
    # # FA_to_T1.inputs.convergence_window_size = [20]*2
    # # FA_to_T1.inputs.smoothing_sigmas = [[1,0], [2,1,0]]
    # # FA_to_T1.inputs.sigma_units = ['vox'] * 2
    # # FA_to_T1.inputs.shrink_factors = [[2,1], [3,2,1]]
    # # FA_to_T1.inputs.use_estimate_learning_rate_once = [True, True]
    # # FA_to_T1.inputs.use_histogram_matching = [True, True] # This is the default
    # # FA_to_T1.inputs.output_warped_image = 'FA_T1space.nii.gz'
    # # FA_to_T1.inputs.num_threads = 2

    #WM_to_FA = pe.Node(interface=ants.ApplyTransforms(), name="WM_to_FA")

    fsl2mrtrix = pe.Node(interface=mrtrix.FSL2MRTrix(), name='fsl2mrtrix')
    fsl2mrtrix.inputs.invert_y = True

    erode_mask_firstpass = pe.Node(interface=mrtrix.Erode(),
                                   name='erode_mask_firstpass')
    erode_mask_firstpass.inputs.out_filename = "b0_mask_median3D_erode.nii.gz"
    erode_mask_secondpass = pe.Node(interface=mrtrix.Erode(),
                                    name='erode_mask_secondpass')
    erode_mask_secondpass.inputs.out_filename = "b0_mask_median3D_erode_secondpass.nii.gz"

    threshold_FA = pe.Node(interface=fsl.ImageMaths(), name='threshold_FA')
    threshold_FA.inputs.op_string = "-thr 0.8 -uthr 0.99"


    get_wm_mask = pe.Node(interface=fsl.ImageMaths(), name='get_wm_mask')
    get_wm_mask.inputs.op_string = "-thr 0.3"

    MRmultiply = pe.Node(interface=mrtrix.MRMultiply(), name='MRmultiply')
    MRmultiply.inputs.out_filename = "Eroded_FA.nii.gz"
    MRmult_merge = pe.Node(interface=util.Merge(2), name='MRmultiply_merge')

    median3d = pe.Node(interface=mrtrix.MedianFilter3D(), name='median3D')

    fdgpet_regions = pe.Node(interface=RegionalValues(), name='fdgpet_regions')

    csdeconv = pe.Node(interface=mrtrix.ConstrainedSphericalDeconvolution(),
                       name='csdeconv')

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

    CSDstreamtrack.inputs.desired_number_of_tracks = 30000

    tck2trk = pe.Node(interface=mrtrix.MRTrix2TrackVis(), name='tck2trk')

    extract_PreCoTh_interface = Function(input_names=["in_file"],
                                         output_names=["out_file"],
                                         function=extract_PreCoTh)

    thalamus2precuneus2cortex_ROIs = pe.Node(
        interface=extract_PreCoTh_interface, name='thalamus2precuneus2cortex_ROIs')

    thalamus2precuneus2cortex = pe.Node(
        interface=cmtk.CreateMatrix(), name="thalamus2precuneus2cortex")
    thalamus2precuneus2cortex.inputs.count_region_intersections = True

    FreeSurferSource = pe.Node(
        interface=nio.FreeSurferSource(), name='fssource')
    mri_convert_Brain = pe.Node(
        interface=fs.MRIConvert(), name='mri_convert_Brain')
    mri_convert_Brain.inputs.out_type = 'niigz'

    mri_convert_WhiteMatter = mri_convert_Brain.clone("mri_convert_WhiteMatter")
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
    workflow.connect(
        [(FreeSurferSource, mri_convert_WhiteMatter, [('wm', 'in_file')])])

    workflow.connect([(inputnode, fsl2mrtrix, [("bvecs", "bvec_file"),
                                               ("bvals", "bval_file")])])

    workflow.connect(inputnode, 'dwi', nonlinfit_node, 'dwi')
    workflow.connect(inputnode, 'subject_id', nonlinfit_node, 'base_name')
    workflow.connect(inputnode, 'bvecs', nonlinfit_node, 'bvecs')
    workflow.connect(inputnode, 'bvals', nonlinfit_node, 'bvals')

    workflow.connect([(inputnode, reslice_fdgpet, [("fdgpet", "in_file")])])
    workflow.connect(
        [(mri_convert_ROIs, reslice_fdgpet, [("out_file", "reslice_like")])])
    workflow.connect(
        [(reslice_fdgpet, fdgpet_regions, [("out_file", "in_files")])])
    workflow.connect(
        [(thalamus2precuneus2cortex_ROIs, fdgpet_regions, [("out_file", "segmentation_file")])])


    #workflow.connect([(nonlinfit_node, FA_to_T1,[('FA','moving_image')])])
    #workflow.connect([(mri_convert_Brain, FA_to_T1,[('out_file','fixed_image')])])
    #workflow.connect([(FA_to_T1, WM_to_FA,[('inverse_composite_transform','transforms')])])

    workflow.connect([(nonlinfit_node, coregister,[("FA","in_file")])])
    workflow.connect([(mri_convert_Brain, coregister,[('out_file','reference')])])
    workflow.connect([(nonlinfit_node, tck2trk,[("FA","image_file")])])
    workflow.connect([(mri_convert_Brain, tck2trk,[("out_file","registration_image_file")])])
    workflow.connect([(coregister, tck2trk,[("out_matrix_file","matrix_file")])])

    #workflow.connect([(coregister, invertxfm,[("out_matrix_file","in_file")])])
    #workflow.connect([(invertxfm, WM_to_FA,[("out_file","in_matrix_file")])])
    #workflow.connect([(mri_convert_WhiteMatter, WM_to_FA,[("out_file","in_file")])])
    #workflow.connect([(nonlinfit_node, WM_to_FA,[("FA","reference")])])

    workflow.connect([(nonlinfit_node, get_wm_mask, [("FA", "in_file")])])
    #workflow.connect([(mri_convert_WhiteMatter, WM_to_FA,[('out_file','input_image')])])
    #workflow.connect([(nonlinfit_node, WM_to_FA,[('FA','reference_image')])])

    workflow.connect([(nonlinfit_node, median3d, [("binary_mask", "in_file")])])
    workflow.connect(
        [(median3d, erode_mask_firstpass, [("out_file", "in_file")])])
    workflow.connect(
        [(erode_mask_firstpass, erode_mask_secondpass, [("out_file", "in_file")])])

    workflow.connect([(nonlinfit_node, MRmult_merge, [("FA", "in1")])])
    workflow.connect(
        [(erode_mask_secondpass, MRmult_merge, [("out_file", "in2")])])
    workflow.connect([(MRmult_merge, MRmultiply, [("out", "in_files")])])
    workflow.connect([(MRmultiply, threshold_FA, [("out_file", "in_file")])])
    workflow.connect(
        [(threshold_FA, estimateresponse, [("out_file", "mask_image")])])

    workflow.connect([(inputnode, estimateresponse, [("dwi", "in_file")])])
    workflow.connect(
        [(fsl2mrtrix, estimateresponse, [("encoding_file", "encoding_file")])])

    workflow.connect([(inputnode, csdeconv, [("dwi", "in_file")])])
    workflow.connect(
        [(get_wm_mask, csdeconv, [("out_file", "mask_image")])])
    workflow.connect(
        [(estimateresponse, csdeconv, [("response", "response_file")])])
    workflow.connect(
        [(fsl2mrtrix, csdeconv, [("encoding_file", "encoding_file")])])

    workflow.connect(
        [(get_wm_mask, CSDstreamtrack, [("out_file", "seed_file")])])
    workflow.connect(
        [(csdeconv, CSDstreamtrack, [("spherical_harmonics_image", "in_file")])])

    workflow.connect([(CSDstreamtrack, tck2trk, [("tracked", "in_file")])])

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

    output_fields = ["fa", "rgb_fa", "md", "csdeconv", "tracts_tck", "rois", "t1_brain", "wmmask_dtispace", "fa_t1space"]

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=output_fields),
        name="outputnode")

    workflow.connect(
        [(CSDstreamtrack, outputnode, [("tracked", "tracts_tck")]),
         (csdeconv, outputnode,
          [("spherical_harmonics_image", "csdeconv")]),
         (nonlinfit_node, outputnode, [("FA", "fa")]),
         (coregister, outputnode, [("out_file", "fa_t1space")]),
         #(WM_to_FA, outputnode, [("out_file", "wmmask_dtispace")]),
         (mri_convert_Brain, outputnode, [("out_file", "t1_brain")]),
         (thalamus2precuneus2cortex_ROIs, outputnode, [("out_file", "rois")]),
         (nonlinfit_node, outputnode, [("rgb_fa", "rgb_fa")]),
         (nonlinfit_node, outputnode, [("MD", "md")])])

    return workflow
