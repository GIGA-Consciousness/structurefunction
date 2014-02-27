import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.mrtrix as mrtrix
import nipype.interfaces.cmtk as cmtk
from nipype.workflows.misc.utils import select_aparc

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from coma.interfaces import RegionalValues, nonlinfit_fn, CMR_glucose


def select_ribbon(list_of_files):
    for idx, in_file in enumerate(list_of_files):
        if in_file == 'ribbon.mgz':
            idx = list_of_files.index(in_file)
    return list_of_files[idx]



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

    dwi = nx.to_numpy_matrix(dwi_ntwk, weight="number_of_fibers")

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
    all_data = f_avg + f_max + f_min + f_std
    all_data = all_data + l_th + r_th + l_cor + r_cor + l_prec

    out_file = op.abspath(subject_id + "_precoth.csv")
    f = open(out_file, "w")
    title_str = ",".join(all_titles) + "\n"
    f.write(title_str)
    data_str = subject_id + "," + ",".join(format(x, "10.5f") for x in all_data) + "\n"
    f.write(data_str)
    f.close()
    return out_file


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

    CSDstreamtrack.inputs.desired_number_of_tracks = 10000

    tck2trk = pe.Node(interface=mrtrix.MRTrix2TrackVis(), name='tck2trk')

    extract_PreCoTh_interface = util.Function(input_names=["in_file"],
                                         output_names=["out_file"],
                                         function=extract_PreCoTh)
    thalamus2precuneus2cortex_ROIs = pe.Node(
        interface=extract_PreCoTh_interface, name='thalamus2precuneus2cortex_ROIs')

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

    if reg_pet_T1:
        reg_pet_T1 = pe.Node(interface=fsl.FLIRT(dof=6), name = 'reg_pet_T1')
        reg_pet_T1.inputs.cost = ('corratio')
    
    reslice_fdgpet = mri_convert_Brain.clone("reslice_fdgpet")

    mri_convert_WhiteMatter = mri_convert_Brain.clone("mri_convert_WhiteMatter")
    mri_convert_Ribbon = mri_convert_Brain.clone("mri_convert_Ribbon")
    mri_convert_ROIs = mri_convert_Brain.clone("mri_convert_ROIs")

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
    workflow.connect(
        [(FreeSurferSource, mri_convert_Ribbon, [(('ribbon', select_ribbon), 'in_file')])])
    workflow.connect(
        [(mri_convert_Ribbon, make_termination_mask, [('out_file', 'in_file')])])

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
            [(mri_convert_Brain, reg_pet_T1, [("out_file", "reference")])])
        workflow.connect(
            [(reg_pet_T1, reslice_fdgpet, [("out_file", "in_file")])])
        workflow.connect(
            [(mri_convert_ROIs, reslice_fdgpet, [("out_file", "reslice_like")])])
        workflow.connect(
            [(reslice_fdgpet, compute_cmr_glc, [("out_file", "in_file")])])
    else:
        workflow.connect([(inputnode, reslice_fdgpet, [("fdgpet", "in_file")])])
        workflow.connect(
            [(mri_convert_ROIs, reslice_fdgpet, [("out_file", "reslice_like")])])
        workflow.connect(
            [(reslice_fdgpet, compute_cmr_glc, [("out_file", "in_file")])])
    workflow.connect(
        [(compute_cmr_glc, fdgpet_regions, [("out_file", "in_files")])])
    workflow.connect(
        [(thalamus2precuneus2cortex_ROIs, fdgpet_regions, [("out_file", "segmentation_file")])])


    #workflow.connect([(nonlinfit_node, FA_to_T1,[('FA','moving_image')])])
    #workflow.connect([(mri_convert_Brain, FA_to_T1,[('out_file','fixed_image')])])
    #workflow.connect([(FA_to_T1, WM_to_FA,[('inverse_composite_transform','transforms')])])

    workflow.connect([(nonlinfit_node, coregister,[("FA","in_file")])])
    workflow.connect([(mri_convert_WhiteMatter, coregister,[('out_file','reference')])])
    workflow.connect([(nonlinfit_node, tck2trk,[("FA","image_file")])])
    workflow.connect([(mri_convert_Brain, tck2trk,[("out_file","registration_image_file")])])
    workflow.connect([(coregister, tck2trk,[("out_matrix_file","matrix_file")])])

    workflow.connect([(coregister, invertxfm,[("out_matrix_file","in_file")])])
    workflow.connect([(invertxfm, WM_to_FA,[("out_file","in_matrix_file")])])
    workflow.connect([(mri_convert_WhiteMatter, WM_to_FA,[("out_file","in_file")])])
    workflow.connect([(nonlinfit_node, WM_to_FA,[("FA","reference")])])
    
    workflow.connect([(invertxfm, TermMask_to_FA,[("out_file","in_matrix_file")])])
    workflow.connect([(make_termination_mask, TermMask_to_FA,[("out_file","in_file")])])
    workflow.connect([(nonlinfit_node, TermMask_to_FA,[("FA","reference")])])


    #workflow.connect([(nonlinfit_node, get_wm_mask, [("FA", "in_file")])])
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
        [(TermMask_to_FA, csdeconv, [("out_file", "mask_image")])])
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
        [(mri_convert_ROIs, thalamus2precuneus2cortex_ROIs, [("out_file", "in_file")])])
    workflow.connect(
        [(thalamus2precuneus2cortex_ROIs, thalamus2precuneus2cortex, [("out_file", "roi_file")])])

    workflow.connect(
        [(inputnode, write_csv_data, [("subject_id", "subject_id")])])
    workflow.connect(
        [(fdgpet_regions, write_csv_data, [("stats_file", "fdg_stats_file")])])
    workflow.connect(
        [(thalamus2precuneus2cortex, write_csv_data, [("matrix_file", "dwi_network_file")])])

    output_fields = ["fa", "rgb_fa", "md", "csdeconv", "tracts_tck", "rois",
        "t1_brain", "wmmask_dtispace", "fa_t1space", "summary", "filtered_tractographies"]

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
         (thalamus2precuneus2cortex, outputnode, [("filtered_tractographies", "filtered_tractographies")]),
         (nonlinfit_node, outputnode, [("rgb_fa", "rgb_fa")]),
         (nonlinfit_node, outputnode, [("MD", "md")]),
         (write_csv_data, outputnode, [("out_file", "summary")]),
         ])

    return workflow


def return_subject_data(subject_id, data_file):
    import csv
    from nipype import logging
    iflogger = logging.getLogger('interface')

    f = open(data_file, 'r')
    csv_line_by_line = csv.reader(f)
    found = False
    # Must be stored in this order!:
    # 'subject_id', 'dose', 'weight', 'delay', 'glycemie', 'scan_time']
    for line in csv_line_by_line:
        if line[0] == subject_id:
            dose, weight, delay, glycemie, scan_time = [float(x) for x in line[1:]]
            iflogger.info('Subject %s found' % subject_id)
            iflogger.info('Dose: %s' % dose)
            iflogger.info('Weight: %s' % weight)
            iflogger.info('Delay: %s' % delay)
            iflogger.info('Glycemie: %s' % glycemie)
            iflogger.info('Scan Time: %s' % scan_time)
            found = True
            break
    if not found:
        raise Exception("Subject id %s was not in the data file!" % subject_id)
    return dose, weight, delay, glycemie, scan_time