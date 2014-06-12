import os
import os.path as op
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.mrtrix as mrtrix
import nipype.interfaces.cmtk as cmtk
from nipype.workflows.misc.utils import select_aparc

fsl.FSLCommand.set_default_output_type('NIFTI')

from coma.interfaces import RegionalValues, nonlinfit_fn, CMR_glucose

def summarize_precoth(dwi_network_file, fdg_stats_file, subject_id):
    import os.path as op
    import scipy.io as sio
    import networkx as nx

    fdg = sio.loadmat(fdg_stats_file)
    dwi_ntwk = nx.read_gpickle(dwi_network_file)

    # Thal L-1 R-2
    # Cortex 3 and 4
    # Prec L-5 R-6
    titles = ["subjid"]
    fdg_avg = ["LTh_CMR_avg","RTh_CMR_avg","LCo_CMR_avg","RCo_CMR_avg","LPre_CMR_avg","RPre_CMR_avg"]
    f_avg = [fdg["func_mean"][0][0],fdg["func_mean"][1][0],fdg["func_mean"][2][0],
               fdg["func_mean"][3][0],fdg["func_mean"][4][0],fdg["func_mean"][5][0]]

    fdg_max = ["LTh_CMR_max","RTh_CMR_max","LCo_CMR_max","RCo_CMR_max","LPre_CMR_max","RPre_CMR_max"]
    f_max = [fdg["func_max"][0][0],fdg["func_max"][1][0],fdg["func_max"][2][0],
               fdg["func_max"][3][0],fdg["func_max"][4][0],fdg["func_max"][5][0]]

    fdg_min = ["LTh_CMR_min","RTh_CMR_min","LCo_CMR_min","RCo_CMR_min","LPre_CMR_min","RPre_CMR_min"]
    f_min = [fdg["func_min"][0][0],fdg["func_min"][1][0],fdg["func_min"][2][0],
               fdg["func_min"][3][0],fdg["func_min"][4][0],fdg["func_min"][5][0]]

    fdg_std = ["LTh_CMR_std","RTh_CMR_std","LCo_CMR_std","RCo_CMR_std","LPre_CMR_std","RPre_CMR_std"]
    f_std = [fdg["func_stdev"][0][0],fdg["func_stdev"][1][0],fdg["func_stdev"][2][0],
               fdg["func_stdev"][3][0],fdg["func_stdev"][4][0],fdg["func_stdev"][5][0]]

    fdg_titles = fdg_avg + fdg_max + fdg_min + fdg_std

    dwi = nx.to_numpy_matrix(dwi_ntwk, weight="weight")

    l_thal = ["LTh_RTh","LTh_LCo","LTh_RCo","LTh_LPre","LTh_RPre"]
    l_th   = [dwi[0,1], dwi[0,2], dwi[0,3], dwi[0,4], dwi[0,5]]
    r_thal = ["RTh_LCo","RTh_RCo","RTh_LPre","RTh_RPre"]
    r_th   = [dwi[1,2], dwi[1,3], dwi[1,4], dwi[1,5]]
    l_co   = ["LCo_RCo","LCo_LPre","LCo_RPre"]
    l_cor  = [dwi[2,3], dwi[2,4], dwi[2,5]]
    r_co   = ["RCo_LPre","RCo_RPre"]
    r_cor  = [dwi[3,4], dwi[3,5]]
    l_pre  = ["LPre_RPre"]
    l_prec = [dwi[4,5]]
    conn_titles = l_thal + r_thal + l_co + r_co + l_pre

    all_titles = titles + fdg_titles + conn_titles
    volume_titles = ["VoxLTh","VoxRTh","VoxLCo", "VoxRCo", "VoxLPre", "VoxRPre"]
    all_titles = all_titles + volume_titles
    volumes = fdg["number_of_voxels"]

    all_data = f_avg + f_max + f_min + f_std + l_th + r_th + l_cor + r_cor + l_prec + volumes[:,0].tolist()

    out_file = op.abspath(subject_id + "_precoth.csv")
    f = open(out_file, "w")
    title_str = ",".join(all_titles) + "\n"
    f.write(title_str)
    all_data = map(float, all_data)
    data_str = subject_id + "," + ",".join(format(x, "10.5f") for x in all_data) + "\n"
    f.write(data_str)
    f.close()
    return out_file

def extract_PreCoTh(in_file, out_filename):
    from nipype.utils.filemanip import split_filename
    import nibabel as nb
    import numpy as np
    import os.path as op
    in_image = nb.load(in_file)
    in_header = in_image.get_header()
    in_data = in_image.get_data()

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
        niiGM[in_data == ma[1]] = ma[0]

    _, name, _ = split_filename(in_file)
    out_file = op.abspath(out_filename)
    try:
        out_image = nb.Nifti1Image(
            data=niiGM, header=in_header, affine=in_image.get_affine())
    except TypeError:
        out_image = nb.Nifti1Image(
            dataobj=niiGM, header=in_header, affine=in_image.get_affine())
    nb.save(out_image, out_file)
    return out_file


def create_precoth_pipeline(name="precoth", tractography_type='probabilistic', reg_pet_T1=True):
    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=["subjects_dir",
                                                 "subject_id",
                                                 "dwi",
                                                 "bvecs",
                                                 "bvals",
                                                 "fdgpet",
                                                 "dose",
                                                 "weight",
                                                 "delay",
                                                 "glycemie",
                                                 "scan_time"]),
        name="inputnode")

    nonlinfit_interface = util.Function(input_names=["dwi", "bvecs", "bvals", "base_name"],
    output_names=["tensor", "FA", "MD", "evecs", "evals", "rgb_fa", "norm", "mode", "binary_mask", "b0_masked"], function=nonlinfit_fn)

    nonlinfit_node = pe.Node(interface=nonlinfit_interface, name="nonlinfit_node")

    coregister = pe.Node(interface=fsl.FLIRT(dof=12), name = 'coregister')
    coregister.inputs.cost = ('normmi')

    invertxfm = pe.Node(interface=fsl.ConvertXFM(), name = 'invertxfm')
    invertxfm.inputs.invert_xfm = True

    WM_to_FA = pe.Node(interface=fsl.ApplyXfm(), name = 'WM_to_FA')
    WM_to_FA.inputs.interp = 'nearestneighbour'
    TermMask_to_FA = WM_to_FA.clone("TermMask_to_FA")

    mni_for_reg = op.join(os.environ["FSL_DIR"],"data","standard","MNI152_T1_1mm.nii.gz")
    reorientBrain = pe.Node(interface=fsl.FLIRT(dof=6), name = 'reorientBrain')
    reorientBrain.inputs.reference = mni_for_reg
    reorientROIs = pe.Node(interface=fsl.ApplyXfm(), name = 'reorientROIs')
    reorientROIs.inputs.interp = "nearestneighbour"
    reorientROIs.inputs.reference = mni_for_reg
    reorientRibbon = reorientROIs.clone("reorientRibbon")
    reorientRibbon.inputs.interp = "nearestneighbour"
    reorientT1 = reorientROIs.clone("reorientT1")
    reorientT1.inputs.interp = "trilinear"

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
    threshold_mode = pe.Node(interface=fsl.ImageMaths(), name='threshold_mode')
    threshold_mode.inputs.op_string = "-thr 0.1 -uthr 0.99"    

    make_termination_mask = pe.Node(interface=fsl.ImageMaths(), name='make_termination_mask')
    make_termination_mask.inputs.op_string = "-bin"

    get_wm_mask = pe.Node(interface=fsl.ImageMaths(), name='get_wm_mask')
    get_wm_mask.inputs.op_string = "-thr 0.1"

    MRmultiply = pe.Node(interface=mrtrix.MRMultiply(), name='MRmultiply')
    MRmultiply.inputs.out_filename = "Eroded_FA.nii.gz"
    MRmult_merge = pe.Node(interface=util.Merge(2), name='MRmultiply_merge')

    median3d = pe.Node(interface=mrtrix.MedianFilter3D(), name='median3D')

    fdgpet_regions = pe.Node(interface=RegionalValues(), name='fdgpet_regions')

    compute_cmr_glc_interface = util.Function(input_names=["in_file", "dose", "weight", "delay",
        "glycemie", "scan_time"], output_names=["out_file"], function=CMR_glucose)
    compute_cmr_glc = pe.Node(interface=compute_cmr_glc_interface, name='compute_cmr_glc')

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

    #CSDstreamtrack.inputs.desired_number_of_tracks = 10000
    CSDstreamtrack.inputs.minimum_tract_length = 50

    tck2trk = pe.Node(interface=mrtrix.MRTrix2TrackVis(), name='tck2trk')

    extract_PreCoTh_interface = util.Function(input_names=["in_file", "out_filename"],
                                         output_names=["out_file"],
                                         function=extract_PreCoTh)
    thalamus2precuneus2cortex_ROIs = pe.Node(
        interface=extract_PreCoTh_interface, name='thalamus2precuneus2cortex_ROIs')


    wm_mask_interface = util.Function(input_names=["in_file", "out_filename"],
                                         output_names=["out_file"],
                                         function=wm_labels_only)
    make_wm_mask = pe.Node(
        interface=wm_mask_interface, name='make_wm_mask')

    write_precoth_data_interface = util.Function(input_names=["dwi_network_file", "fdg_stats_file", "subject_id"],
                                         output_names=["out_file"],
                                         function=summarize_precoth)
    write_csv_data = pe.Node(
        interface=write_precoth_data_interface, name='write_csv_data')

    thalamus2precuneus2cortex = pe.Node(
        interface=cmtk.CreateMatrix(), name="thalamus2precuneus2cortex")
    thalamus2precuneus2cortex.inputs.count_region_intersections = True

    FreeSurferSource = pe.Node(
        interface=nio.FreeSurferSource(), name='fssource')
    mri_convert_Brain = pe.Node(
        interface=fs.MRIConvert(), name='mri_convert_Brain')
    mri_convert_Brain.inputs.out_type = 'niigz'
    mri_convert_Brain.inputs.no_change = True

    if reg_pet_T1:
        reg_pet_T1 = pe.Node(interface=fsl.FLIRT(dof=6), name = 'reg_pet_T1')
        reg_pet_T1.inputs.cost = ('corratio')
    
    reslice_fdgpet = mri_convert_Brain.clone("reslice_fdgpet")
    reslice_fdgpet.inputs.no_change = True

    mri_convert_Ribbon = mri_convert_Brain.clone("mri_convert_Ribbon")
    mri_convert_ROIs = mri_convert_Brain.clone("mri_convert_ROIs")
    mri_convert_T1 = mri_convert_Brain.clone("mri_convert_T1")

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    workflow.connect(
        [(inputnode, FreeSurferSource, [("subjects_dir", "subjects_dir")])])
    workflow.connect(
        [(inputnode, FreeSurferSource, [("subject_id", "subject_id")])])

    workflow.connect(
        [(FreeSurferSource, mri_convert_T1, [('T1', 'in_file')])])
    workflow.connect(
        [(mri_convert_T1, reorientT1, [('out_file', 'in_file')])])

    workflow.connect(
        [(FreeSurferSource, mri_convert_Brain, [('brain', 'in_file')])])
    workflow.connect(
        [(mri_convert_Brain, reorientBrain, [('out_file', 'in_file')])])
    workflow.connect(
        [(reorientBrain, reorientROIs, [('out_matrix_file', 'in_matrix_file')])])
    workflow.connect(
        [(reorientBrain, reorientRibbon, [('out_matrix_file', 'in_matrix_file')])])
    workflow.connect(
        [(reorientBrain, reorientT1, [('out_matrix_file', 'in_matrix_file')])])

    workflow.connect(
        [(FreeSurferSource, mri_convert_ROIs, [(('aparc_aseg', select_aparc), 'in_file')])])
    workflow.connect(
        [(mri_convert_ROIs, reorientROIs, [('out_file', 'in_file')])])
    workflow.connect(
        [(reorientROIs, make_wm_mask, [('out_file', 'in_file')])])

    workflow.connect(
        [(FreeSurferSource, mri_convert_Ribbon, [(('ribbon', select_ribbon), 'in_file')])])
    workflow.connect(
        [(mri_convert_Ribbon, reorientRibbon, [('out_file', 'in_file')])])
    workflow.connect(
        [(reorientRibbon, make_termination_mask, [('out_file', 'in_file')])])

    workflow.connect([(inputnode, fsl2mrtrix, [("bvecs", "bvec_file"),
                                               ("bvals", "bval_file")])])

    workflow.connect(inputnode, 'dwi', nonlinfit_node, 'dwi')
    workflow.connect(inputnode, 'subject_id', nonlinfit_node, 'base_name')
    workflow.connect(inputnode, 'bvecs', nonlinfit_node, 'bvecs')
    workflow.connect(inputnode, 'bvals', nonlinfit_node, 'bvals')

    workflow.connect([(inputnode, compute_cmr_glc, [("dose", "dose")])])
    workflow.connect([(inputnode, compute_cmr_glc, [("weight", "weight")])])
    workflow.connect([(inputnode, compute_cmr_glc, [("delay", "delay")])])
    workflow.connect([(inputnode, compute_cmr_glc, [("glycemie", "glycemie")])])
    workflow.connect([(inputnode, compute_cmr_glc, [("scan_time", "scan_time")])])

    if reg_pet_T1:
        workflow.connect([(inputnode, reg_pet_T1, [("fdgpet", "in_file")])])
        workflow.connect(
            [(reorientBrain, reg_pet_T1, [("out_file", "reference")])])
        workflow.connect(
            [(reg_pet_T1, reslice_fdgpet, [("out_file", "in_file")])])
        workflow.connect(
            [(reorientROIs, reslice_fdgpet, [("out_file", "reslice_like")])])
        workflow.connect(
            [(reslice_fdgpet, compute_cmr_glc, [("out_file", "in_file")])])
    else:
        workflow.connect([(inputnode, reslice_fdgpet, [("fdgpet", "in_file")])])
        workflow.connect(
            [(reorientROIs, reslice_fdgpet, [("out_file", "reslice_like")])])
        workflow.connect(
            [(reslice_fdgpet, compute_cmr_glc, [("out_file", "in_file")])])
    workflow.connect(
        [(compute_cmr_glc, fdgpet_regions, [("out_file", "in_files")])])
    workflow.connect(
        [(thalamus2precuneus2cortex_ROIs, fdgpet_regions, [("out_file", "segmentation_file")])])

    workflow.connect([(nonlinfit_node, coregister,[("FA","in_file")])])
    workflow.connect([(make_wm_mask, coregister,[('out_file','reference')])])
    workflow.connect([(nonlinfit_node, tck2trk,[("FA","image_file")])])
    workflow.connect([(reorientBrain, tck2trk,[("out_file","registration_image_file")])])
    workflow.connect([(coregister, tck2trk,[("out_matrix_file","matrix_file")])])

    workflow.connect([(coregister, invertxfm,[("out_matrix_file","in_file")])])
    workflow.connect([(invertxfm, WM_to_FA,[("out_file","in_matrix_file")])])
    workflow.connect([(make_wm_mask, WM_to_FA,[("out_file","in_file")])])
    workflow.connect([(nonlinfit_node, WM_to_FA,[("FA","reference")])])
    
    workflow.connect([(invertxfm, TermMask_to_FA,[("out_file","in_matrix_file")])])
    workflow.connect([(make_termination_mask, TermMask_to_FA,[("out_file","in_file")])])
    workflow.connect([(nonlinfit_node, TermMask_to_FA,[("FA","reference")])])

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
    #workflow.connect(
    #    [(TermMask_to_FA, csdeconv, [("out_file", "mask_image")])])
    workflow.connect(
        [(estimateresponse, csdeconv, [("response", "response_file")])])
    workflow.connect(
        [(fsl2mrtrix, csdeconv, [("encoding_file", "encoding_file")])])
    workflow.connect(
        [(WM_to_FA, CSDstreamtrack, [("out_file", "seed_file")])])
    workflow.connect(
        [(TermMask_to_FA, CSDstreamtrack, [("out_file", "mask_file")])])
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
        [(reorientROIs, thalamus2precuneus2cortex_ROIs, [("out_file", "in_file")])])
    workflow.connect(
        [(thalamus2precuneus2cortex_ROIs, thalamus2precuneus2cortex, [("out_file", "roi_file")])])
    workflow.connect(
        [(thalamus2precuneus2cortex, fdgpet_regions, [("matrix_file", "resolution_network_file")])])

    workflow.connect(
        [(inputnode, write_csv_data, [("subject_id", "subject_id")])])
    workflow.connect(
        [(fdgpet_regions, write_csv_data, [("stats_file", "fdg_stats_file")])])
    workflow.connect(
        [(thalamus2precuneus2cortex, write_csv_data, [("intersection_matrix_file", "dwi_network_file")])])

    output_fields = ["fa", "rgb_fa", "md", "csdeconv", "tracts_tck", "rois", "t1",
        "t1_brain", "wmmask_dtispace", "fa_t1space", "summary", "filtered_tractographies",
        "matrix_file", "connectome", "CMR_nodes", "fiber_labels_noorphans", "fiber_length_file",
        "fiber_label_file", "intersection_matrix_mat_file"]

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=output_fields),
        name="outputnode")

    workflow.connect(
        [(CSDstreamtrack, outputnode, [("tracked", "tracts_tck")]),
         (csdeconv, outputnode,
          [("spherical_harmonics_image", "csdeconv")]),
         (nonlinfit_node, outputnode, [("FA", "fa")]),
         (coregister, outputnode, [("out_file", "fa_t1space")]),
         (reorientBrain, outputnode, [("out_file", "t1_brain")]),
         (reorientT1, outputnode, [("out_file", "t1")]),
         (thalamus2precuneus2cortex_ROIs, outputnode, [("out_file", "rois")]),
         (thalamus2precuneus2cortex, outputnode, [("filtered_tractographies", "filtered_tractographies")]),
         (thalamus2precuneus2cortex, outputnode, [("matrix_file", "connectome")]),
         (thalamus2precuneus2cortex, outputnode, [("fiber_labels_noorphans", "fiber_labels_noorphans")]),
         (thalamus2precuneus2cortex, outputnode, [("fiber_length_file", "fiber_length_file")]),
         (thalamus2precuneus2cortex, outputnode, [("fiber_label_file", "fiber_label_file")]),
         (thalamus2precuneus2cortex, outputnode, [("intersection_matrix_mat_file", "intersection_matrix_mat_file")]),
         (fdgpet_regions, outputnode, [("networks", "CMR_nodes")]),
         (nonlinfit_node, outputnode, [("rgb_fa", "rgb_fa")]),
         (nonlinfit_node, outputnode, [("MD", "md")]),
         (write_csv_data, outputnode, [("out_file", "summary")]),
         ])

    return workflow


def create_precoth_pipeline_step1(name="precoth_step1", reg_pet_T1=True, auto_reorient=True):
    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=["subjects_dir",
                                                 "subject_id",
                                                 "dwi",
                                                 "bvecs",
                                                 "bvals",
                                                 "fdgpet"]),
        name="inputnode")

    nonlinfit_interface = util.Function(input_names=["dwi", "bvecs", "bvals", "base_name"],
    output_names=["tensor", "FA", "MD", "evecs", "evals", "rgb_fa", "norm", "mode", "binary_mask", "b0_masked"], function=nonlinfit_fn)

    nonlinfit_node = pe.Node(interface=nonlinfit_interface, name="nonlinfit_node")
    erode_mask_firstpass = pe.Node(interface=mrtrix.Erode(),
                                   name='erode_mask_firstpass')
    erode_mask_firstpass.inputs.out_filename = "b0_mask_median3D_erode.nii.gz"
    erode_mask_secondpass = pe.Node(interface=mrtrix.Erode(),
                                    name='erode_mask_secondpass')
    erode_mask_secondpass.inputs.out_filename = "b0_mask_median3D_erode_secondpass.nii.gz"

    threshold_FA = pe.Node(interface=fsl.ImageMaths(), name='threshold_FA')
    threshold_FA.inputs.op_string = "-thr 0.8 -uthr 0.99"
    threshold_mode = pe.Node(interface=fsl.ImageMaths(), name='threshold_mode')
    threshold_mode.inputs.op_string = "-thr 0.9 -fmedian -fmedian"

    make_termination_mask = pe.Node(interface=fsl.ImageMaths(), name='make_termination_mask')
    make_termination_mask.inputs.op_string = "-bin"

    fast_seg_T1 = pe.Node(interface=fsl.FAST(), name='fast_seg_T1')
    fast_seg_T1.inputs.segments = True
    fast_seg_T1.inputs.probability_maps = True

    fix_wm_mask = pe.Node(interface=fsl.MultiImageMaths(), name='fix_wm_mask')
    fix_wm_mask.inputs.op_string = "-mul %s"

    fix_termination_mask = pe.Node(interface=fsl.MultiImageMaths(), name='fix_termination_mask')
    fix_termination_mask.inputs.op_string = "-binv -mul %s"

    wm_mask_interface = util.Function(input_names=["in_file", "out_filename"],
                                         output_names=["out_file"],
                                         function=wm_labels_only)
    make_wm_mask = pe.Node(interface=wm_mask_interface, name='make_wm_mask')

    MRmultiply = pe.Node(interface=mrtrix.MRMultiply(), name='MRmultiply')
    MRmultiply.inputs.out_filename = "Eroded_FA.nii.gz"

    MultFAbyMode = pe.Node(interface=mrtrix.MRMultiply(), name='MultFAbyMode')

    MRmult_merge = pe.Node(interface=util.Merge(2), name='MRmultiply_merge')
    MultFAbyMode_merge = pe.Node(interface=util.Merge(2), name='MultFAbyMode_merge')

    median3d = pe.Node(interface=mrtrix.MedianFilter3D(), name='median3D')

    FreeSurferSource = pe.Node(
        interface=nio.FreeSurferSource(), name='fssource')
    mri_convert_Brain = pe.Node(
        interface=fs.MRIConvert(), name='mri_convert_Brain')
    mri_convert_Brain.inputs.out_type = 'nii'
    mri_convert_Brain.inputs.no_change = True
    
    mri_convert_Ribbon = mri_convert_Brain.clone("mri_convert_Ribbon")
    mri_convert_ROIs = mri_convert_Brain.clone("mri_convert_ROIs")
    mri_convert_T1 = mri_convert_Brain.clone("mri_convert_T1")
    
    mni_for_reg = op.join(os.environ["FSL_DIR"],"data","standard","MNI152_T1_1mm.nii.gz")

    if auto_reorient:
        reorientBrain = pe.Node(interface=fsl.FLIRT(dof=6), name = 'reorientBrain')
        reorientBrain.inputs.reference = mni_for_reg
        reorientROIs = pe.Node(interface=fsl.ApplyXfm(), name = 'reorientROIs')
        reorientROIs.inputs.interp = "nearestneighbour"
        reorientROIs.inputs.reference = mni_for_reg
        reorientRibbon = reorientROIs.clone("reorientRibbon")
        reorientRibbon.inputs.interp = "nearestneighbour"
        reorientT1 = reorientROIs.clone("reorientT1")
        reorientT1.inputs.interp = "trilinear"

    if reg_pet_T1:
        reg_pet_T1 = pe.Node(interface=fsl.FLIRT(dof=6), name = 'reg_pet_T1')
        reg_pet_T1.inputs.cost = ('corratio')
    
    reslice_fdgpet = mri_convert_Brain.clone("reslice_fdgpet")
    reslice_fdgpet.inputs.no_change = True

    extract_PreCoTh_interface = util.Function(input_names=["in_file", "out_filename"],
                                         output_names=["out_file"],
                                         function=extract_PreCoTh)
    thalamus2precuneus2cortex_ROIs = pe.Node(
        interface=extract_PreCoTh_interface, name='thalamus2precuneus2cortex_ROIs')


    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    workflow.connect(
        [(inputnode, FreeSurferSource, [("subjects_dir", "subjects_dir")])])
    workflow.connect(
        [(inputnode, FreeSurferSource, [("subject_id", "subject_id")])])

    workflow.connect(
        [(FreeSurferSource, mri_convert_T1, [('T1', 'in_file')])])
    workflow.connect(
        [(FreeSurferSource, mri_convert_Brain, [('brain', 'in_file')])])


    if auto_reorient:
        workflow.connect(
            [(mri_convert_T1, reorientT1, [('out_file', 'in_file')])])
        workflow.connect(
            [(mri_convert_Brain, reorientBrain, [('out_file', 'in_file')])])
        workflow.connect(
            [(reorientBrain, reorientROIs, [('out_matrix_file', 'in_matrix_file')])])
        workflow.connect(
            [(reorientBrain, reorientRibbon, [('out_matrix_file', 'in_matrix_file')])])
        workflow.connect(
            [(reorientBrain, reorientT1, [('out_matrix_file', 'in_matrix_file')])])

    workflow.connect(
        [(FreeSurferSource, mri_convert_ROIs, [(('aparc_aseg', select_aparc), 'in_file')])])
    
    if auto_reorient:
        workflow.connect(
            [(mri_convert_ROIs, reorientROIs, [('out_file', 'in_file')])])
        workflow.connect(
            [(reorientROIs, make_wm_mask, [('out_file', 'in_file')])])
        workflow.connect(
            [(reorientROIs, thalamus2precuneus2cortex_ROIs, [("out_file", "in_file")])])
    else:
        workflow.connect(
            [(mri_convert_ROIs, make_wm_mask, [('out_file', 'in_file')])])
        workflow.connect(
            [(mri_convert_ROIs, thalamus2precuneus2cortex_ROIs, [("out_file", "in_file")])])

    workflow.connect(
        [(FreeSurferSource, mri_convert_Ribbon, [(('ribbon', select_ribbon), 'in_file')])])
    if auto_reorient:
        workflow.connect(
            [(mri_convert_Ribbon, reorientRibbon, [('out_file', 'in_file')])])
        workflow.connect(
            [(reorientRibbon, make_termination_mask, [('out_file', 'in_file')])])
    else:
        workflow.connect(
            [(mri_convert_Ribbon, make_termination_mask, [('out_file', 'in_file')])])        

    if auto_reorient:
        workflow.connect(
            [(reorientBrain, fast_seg_T1, [('out_file', 'in_files')])])
    else:
        workflow.connect(
            [(mri_convert_Brain, fast_seg_T1, [('out_file', 'in_files')])])


    workflow.connect(
        [(inputnode, fast_seg_T1, [("subject_id", "out_basename")])])
    workflow.connect([(fast_seg_T1, fix_termination_mask, [(('tissue_class_files', select_CSF), 'in_file')])])
    workflow.connect([(fast_seg_T1, fix_wm_mask, [(('tissue_class_files', select_WM), 'in_file')])])


    workflow.connect(
        [(make_termination_mask, fix_termination_mask, [('out_file', 'operand_files')])])
    workflow.connect(
        [(make_wm_mask, fix_wm_mask, [('out_file', 'operand_files')])])

    workflow.connect(inputnode, 'dwi', nonlinfit_node, 'dwi')
    workflow.connect(inputnode, 'subject_id', nonlinfit_node, 'base_name')
    workflow.connect(inputnode, 'bvecs', nonlinfit_node, 'bvecs')
    workflow.connect(inputnode, 'bvals', nonlinfit_node, 'bvals')

    if reg_pet_T1:
        workflow.connect([(inputnode, reg_pet_T1, [("fdgpet", "in_file")])])
        if auto_reorient:
            workflow.connect(
                [(reorientBrain, reg_pet_T1, [("out_file", "reference")])])
            workflow.connect(
                [(reorientROIs, reslice_fdgpet, [("out_file", "reslice_like")])])
        else:
            workflow.connect(
                [(mri_convert_Brain, reg_pet_T1, [("out_file", "reference")])])
            workflow.connect(
                [(mri_convert_ROIs, reslice_fdgpet, [("out_file", "reslice_like")])])

        workflow.connect(
            [(reg_pet_T1, reslice_fdgpet, [("out_file", "in_file")])])

    else:
        workflow.connect([(inputnode, reslice_fdgpet, [("fdgpet", "in_file")])])
        if auto_reorient:
            workflow.connect(
                [(reorientROIs, reslice_fdgpet, [("out_file", "reslice_like")])])
        else:
            workflow.connect(
                [(mri_convert_ROIs, reslice_fdgpet, [("out_file", "reslice_like")])])

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

    workflow.connect([(nonlinfit_node, threshold_mode, [("mode", "in_file")])])
    workflow.connect([(threshold_mode, MultFAbyMode_merge, [("out_file", "in1")])])
    workflow.connect([(threshold_FA, MultFAbyMode_merge, [("out_file", "in2")])])
    workflow.connect([(MultFAbyMode_merge, MultFAbyMode, [("out", "in_files")])])
    workflow.connect([(inputnode, MultFAbyMode, [(('subject_id', add_subj_name_to_sfmask), 'out_filename')])])

    workflow.connect([(inputnode, reslice_fdgpet, [(('subject_id', add_subj_name_to_fdgpet), 'out_file')])])
    workflow.connect([(inputnode, make_wm_mask, [(('subject_id', add_subj_name_to_wmmask), 'out_filename')])])
    workflow.connect([(inputnode, fix_wm_mask, [(('subject_id', add_subj_name_to_wmmask), 'out_file')])])
    workflow.connect([(inputnode, fix_termination_mask, [(('subject_id', add_subj_name_to_termmask), 'out_file')])])
    workflow.connect([(inputnode, thalamus2precuneus2cortex_ROIs, [(('subject_id', add_subj_name_to_rois), 'out_filename')])])
    if auto_reorient:
        workflow.connect([(inputnode, reorientT1, [(('subject_id', add_subj_name_to_T1), 'out_file')])])
        workflow.connect([(inputnode, reorientBrain, [(('subject_id', add_subj_name_to_T1brain), 'out_file')])])
    else:
        workflow.connect([(inputnode, mri_convert_T1, [(('subject_id', add_subj_name_to_T1), 'out_file')])])
        workflow.connect([(inputnode, mri_convert_Brain, [(('subject_id', add_subj_name_to_T1brain), 'out_file')])])

    output_fields = ["single_fiber_mask", "fa", "rgb_fa", "md", "t1", "t1_brain",
    "wm_mask", "term_mask", "fdgpet", "rois","mode", "tissue_class_files", "probability_maps"]

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=output_fields),
        name="outputnode")

    workflow.connect([(fast_seg_T1, outputnode, [("tissue_class_files", "tissue_class_files")])])
    workflow.connect([(fast_seg_T1, outputnode, [("probability_maps", "probability_maps")])])
    
    workflow.connect([
         (nonlinfit_node, outputnode, [("FA", "fa")]),
         (nonlinfit_node, outputnode, [("rgb_fa", "rgb_fa")]),
         (nonlinfit_node, outputnode, [("MD", "md")]),
         (nonlinfit_node, outputnode, [("mode", "mode")]),
         (MultFAbyMode, outputnode, [("out_file", "single_fiber_mask")]),
         (fix_wm_mask, outputnode, [("out_file", "wm_mask")]),
         (fix_termination_mask, outputnode, [("out_file", "term_mask")]),
         (reslice_fdgpet, outputnode, [("out_file", "fdgpet")]),
         (thalamus2precuneus2cortex_ROIs, outputnode, [("out_file", "rois")]),
         ])

    if auto_reorient:
        workflow.connect([
            (reorientBrain, outputnode, [("out_file", "t1_brain")]),
            (reorientT1, outputnode, [("out_file", "t1")]),
            ])
    else:
        workflow.connect([
            (mri_convert_Brain, outputnode, [("out_file", "t1_brain")]),
            (mri_convert_T1, outputnode, [("out_file", "t1")]),
            ])
    return workflow




def create_precoth_pipeline_step2(name="precoth_step2", tractography_type='probabilistic'):
    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=["subjects_dir",
                                                 "subject_id",
                                                 "dwi",
                                                 "bvecs",
                                                 "bvals",
                                                 "fdgpet",
                                                 "dose",
                                                 "weight",
                                                 "delay",
                                                 "glycemie",
                                                 "scan_time",
                                                 "single_fiber_mask",
                                                 "fa",
                                                 "rgb_fa",
                                                 "md",
                                                 "t1_brain",
                                                 "t1",
                                                 "wm_mask",
                                                 "term_mask",
                                                 "rois",
                                                 ]),
        name="inputnode")

    coregister = pe.Node(interface=fsl.FLIRT(dof=12), name = 'coregister')
    coregister.inputs.cost = ('normmi')

    invertxfm = pe.Node(interface=fsl.ConvertXFM(), name = 'invertxfm')
    invertxfm.inputs.invert_xfm = True

    WM_to_FA = pe.Node(interface=fsl.ApplyXfm(), name = 'WM_to_FA')
    WM_to_FA.inputs.interp = 'nearestneighbour'
    TermMask_to_FA = WM_to_FA.clone("TermMask_to_FA")

    rgb_fa_t1space = pe.Node(interface=fsl.ApplyXfm(), name = 'rgb_fa_t1space')
    md_to_T1 = pe.Node(interface=fsl.ApplyXfm(), name = 'md_to_T1')

    t1_dtispace = pe.Node(interface=fsl.ApplyXfm(), name = 't1_dtispace')

    fsl2mrtrix = pe.Node(interface=mrtrix.FSL2MRTrix(), name='fsl2mrtrix')
    fsl2mrtrix.inputs.invert_y = True

    fdgpet_regions = pe.Node(interface=RegionalValues(), name='fdgpet_regions')

    compute_cmr_glc_interface = util.Function(input_names=["in_file", "dose", "weight", "delay",
        "glycemie", "scan_time"], output_names=["out_file"], function=CMR_glucose)
    compute_cmr_glc = pe.Node(interface=compute_cmr_glc_interface, name='compute_cmr_glc')

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

    CSDstreamtrack.inputs.minimum_tract_length = 50

    CSDstreamtrack.inputs.desired_number_of_tracks = 10000

    tck2trk = pe.Node(interface=mrtrix.MRTrix2TrackVis(), name='tck2trk')

    write_precoth_data_interface = util.Function(input_names=["dwi_network_file", "fdg_stats_file", "subject_id"],
                                         output_names=["out_file"],
                                         function=summarize_precoth)
    write_csv_data = pe.Node(
        interface=write_precoth_data_interface, name='write_csv_data')

    thalamus2precuneus2cortex = pe.Node(
        interface=cmtk.CreateMatrix(), name="thalamus2precuneus2cortex")
    thalamus2precuneus2cortex.inputs.count_region_intersections = True

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    workflow.connect([(inputnode, fsl2mrtrix, [("bvecs", "bvec_file"),
                                               ("bvals", "bval_file")])])

    workflow.connect([(inputnode, fdgpet_regions, [("rois", "segmentation_file")])])
    workflow.connect([(inputnode, compute_cmr_glc, [("fdgpet", "in_file")])])
    workflow.connect([(inputnode, compute_cmr_glc, [("dose", "dose")])])
    workflow.connect([(inputnode, compute_cmr_glc, [("weight", "weight")])])
    workflow.connect([(inputnode, compute_cmr_glc, [("delay", "delay")])])
    workflow.connect([(inputnode, compute_cmr_glc, [("glycemie", "glycemie")])])
    workflow.connect([(inputnode, compute_cmr_glc, [("scan_time", "scan_time")])])
    workflow.connect([(compute_cmr_glc, fdgpet_regions, [("out_file", "in_files")])])

    workflow.connect([(inputnode, coregister,[("fa","in_file")])])
    workflow.connect([(inputnode, coregister,[('wm_mask','reference')])])
    workflow.connect([(inputnode, tck2trk,[("fa","image_file")])])
    
    workflow.connect([(inputnode, tck2trk,[("wm_mask","registration_image_file")])])
    workflow.connect([(coregister, tck2trk,[("out_matrix_file","matrix_file")])])
    
    workflow.connect([(coregister, invertxfm,[("out_matrix_file","in_file")])])

    workflow.connect([(inputnode, t1_dtispace,[("t1","in_file")])])
    workflow.connect([(invertxfm, t1_dtispace,[("out_file","in_matrix_file")])])
    workflow.connect([(inputnode, t1_dtispace,[("fa","reference")])])

    workflow.connect([(inputnode, rgb_fa_t1space,[("rgb_fa","in_file")])])
    workflow.connect([(coregister, rgb_fa_t1space,[("out_matrix_file","in_matrix_file")])])
    workflow.connect([(inputnode, rgb_fa_t1space,[('wm_mask','reference')])])

    workflow.connect([(inputnode, md_to_T1,[("md","in_file")])])
    workflow.connect([(coregister, md_to_T1,[("out_matrix_file","in_matrix_file")])])
    workflow.connect([(inputnode, md_to_T1,[('wm_mask','reference')])])

    workflow.connect([(invertxfm, WM_to_FA,[("out_file","in_matrix_file")])])
    workflow.connect([(inputnode, WM_to_FA,[("wm_mask","in_file")])])
    workflow.connect([(inputnode, WM_to_FA,[("fa","reference")])])
    
    workflow.connect([(invertxfm, TermMask_to_FA,[("out_file","in_matrix_file")])])
    workflow.connect([(inputnode, TermMask_to_FA,[("term_mask","in_file")])])
    workflow.connect([(inputnode, TermMask_to_FA,[("fa","reference")])])

    workflow.connect([(inputnode, estimateresponse, [("single_fiber_mask", "mask_image")])])

    workflow.connect([(inputnode, estimateresponse, [("dwi", "in_file")])])
    workflow.connect(
        [(fsl2mrtrix, estimateresponse, [("encoding_file", "encoding_file")])])

    workflow.connect([(inputnode, csdeconv, [("dwi", "in_file")])])
    #workflow.connect(
    #    [(TermMask_to_FA, csdeconv, [("out_file", "mask_image")])])
    workflow.connect(
        [(estimateresponse, csdeconv, [("response", "response_file")])])
    workflow.connect(
        [(fsl2mrtrix, csdeconv, [("encoding_file", "encoding_file")])])
    workflow.connect(
        [(WM_to_FA, CSDstreamtrack, [("out_file", "seed_file")])])
    workflow.connect(
        [(TermMask_to_FA, CSDstreamtrack, [("out_file", "mask_file")])])
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
        [(inputnode, thalamus2precuneus2cortex, [("rois", "roi_file")])])
    workflow.connect(
        [(thalamus2precuneus2cortex, fdgpet_regions, [("intersection_matrix_file", "resolution_network_file")])])

    workflow.connect(
        [(inputnode, write_csv_data, [("subject_id", "subject_id")])])
    workflow.connect(
        [(fdgpet_regions, write_csv_data, [("stats_file", "fdg_stats_file")])])
    workflow.connect(
        [(thalamus2precuneus2cortex, write_csv_data, [("intersection_matrix_file", "dwi_network_file")])])

    output_fields = ["csdeconv", "tracts_tck", "summary", "filtered_tractographies",
        "matrix_file", "connectome", "CMR_nodes", "cmr_glucose", "fiber_labels_noorphans", "fiber_length_file",
        "fiber_label_file", "fa_t1space", "rgb_fa_t1space", "md_t1space", "fa_t1xform", "t1_dtispace",
        "intersection_matrix_mat_file", "dti_stats"]

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=output_fields),
        name="outputnode")

    workflow.connect(
        [(CSDstreamtrack, outputnode, [("tracked", "tracts_tck")]),
         (csdeconv, outputnode,
          [("spherical_harmonics_image", "csdeconv")]),
         (coregister, outputnode, [("out_file", "fa_t1space")]),
         (rgb_fa_t1space, outputnode, [("out_file", "rgb_fa_t1space")]),
         (md_to_T1, outputnode, [("out_file", "md_t1space")]),
         (t1_dtispace, outputnode, [("out_file", "t1_dtispace")]),
         (coregister, outputnode, [("out_matrix_file", "fa_t1xform")]),
         (thalamus2precuneus2cortex, outputnode, [("filtered_tractographies", "filtered_tractographies")]),
         (thalamus2precuneus2cortex, outputnode, [("matrix_file", "connectome")]),
         (thalamus2precuneus2cortex, outputnode, [("fiber_labels_noorphans", "fiber_labels_noorphans")]),
         (thalamus2precuneus2cortex, outputnode, [("fiber_length_file", "fiber_length_file")]),
         (thalamus2precuneus2cortex, outputnode, [("fiber_label_file", "fiber_label_file")]),
         (thalamus2precuneus2cortex, outputnode, [("intersection_matrix_mat_file", "intersection_matrix_mat_file")]),
         (thalamus2precuneus2cortex, outputnode, [("stats_file", "dti_stats")]),
         (fdgpet_regions, outputnode, [("networks", "CMR_nodes")]),
         (write_csv_data, outputnode, [("out_file", "summary")]),
         (compute_cmr_glc, outputnode, [("out_file", "cmr_glucose")]),
         ])

    return workflow
