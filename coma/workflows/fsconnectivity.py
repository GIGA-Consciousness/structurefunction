import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs    # freesurfer
import nipype.interfaces.mrtrix as mrtrix
import nipype.interfaces.cmtk as cmtk
import nipype.interfaces.dipy as dipy
import nipype.algorithms.misc as misc
import inspect
import os.path as op                      # system functions
from nipype.workflows.dmri.fsl.epi import create_eddy_correct_pipeline
from nipype.workflows.dmri.connectivity.nx import create_networkx_pipeline, create_cmats_to_csv_pipeline
from nipype.workflows.misc.utils import select_aparc_annot
from coma.workflows.dti.basic import damaged_brain_dti_processing
from coma.workflows.dti.tracking import anatomically_constrained_tracking
from coma.workflows.dmnwf import create_reg_and_label_wf

from coma.helpers import (add_subj_name_to_T1_dwi, add_subj_name_to_nxConnectome, add_subj_name_to_Connectome)


def create_fsconnectivity_pipeline(name="fsconnectivity", manual_seg_rois=False, parcellation_name = "scale33"):
    inputfields = ["subjects_dir",
                     "subject_id",
                     "dwi",
                     "bvecs",
                     "bvals",
                     "resolution_network_file"]

    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=inputfields),
        name="inputnode")

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=[# Outputs from the DWI workflow
                                                 "single_fiber_mask",
                                                 "fa",
                                                 "rgb_fa",
                                                 "md",
                                                 "mode",
                                                 "t1",
                                                 "t1_brain",
                                                 "wm_mask",
                                                 "term_mask",
                                                 "aparc_aseg",
                                                 "tissue_class_files",
                                                 "gm_prob",
                                                 "wm_prob",
                                                 "csf_prob",

                                                 # Outputs from registration and labelling
                                                 "rois_to_dwi",
                                                 "wmmask_to_dwi",
                                                 "termmask_to_dwi",
                                                 "dwi_to_t1_matrix",
                                                 "highres_t1_to_dwi_matrix",

                                                 # T1 in DWI space for reference
                                                 "t1_to_dwi",

                                                 # Outputs from tracking
                                                 "fiber_odfs",
                                                 "fiber_tracks_tck_dwi",
                                                 "fiber_tracks_trk_t1",

                                                 # Outputs from connectivity mapping
                                                 "connectome",
                                                 "nxstatscff",
                                                 "nxmatlab",
                                                 "nxcsv",
                                                 "cmatrix",
                                                 "matrix_file",
                                                 ]),
        name="outputnode")


    t1_to_dwi = pe.Node(interface=fsl.ApplyXfm(),
        name = 't1_to_dwi')

    termmask_to_dwi = t1_to_dwi.clone("termmask_to_dwi")

    dtiproc = damaged_brain_dti_processing("dtiproc", use_FAST_masks=True)
    reg_label = create_reg_and_label_wf("reg_label", manual_seg_rois=True)

    FreeSurferSource = pe.Node(interface=nio.FreeSurferSource(), name='fssource')
    FreeSurferSourceLH = pe.Node(interface=nio.FreeSurferSource(), name='fssourceLH')
    FreeSurferSourceLH.inputs.hemi = 'lh'

    FreeSurferSourceRH = pe.Node(interface=nio.FreeSurferSource(), name='fssourceRH')
    FreeSurferSourceRH.inputs.hemi = 'rh'

    """
    Creating the workflow's nodes
    =============================
    """

    """
    Conversion nodes
    ----------------
    """

    """
    A number of conversion operations are required to obtain NIFTI files from the FreesurferSource for each subject.
    Nodes are used to convert the following:
        * Original structural image to NIFTI
        * Pial, white, inflated, and spherical surfaces for both the left and right hemispheres are converted to GIFTI for visualization in ConnectomeViewer
        * Parcellated annotation files for the left and right hemispheres are also converted to GIFTI

    """

    mri_convert_Brain = pe.Node(interface=fs.MRIConvert(), name='mri_convert_Brain')
    mri_convert_Brain.inputs.out_type = 'nii'
    mri_convert_ROI_scale500 = mri_convert_Brain.clone('mri_convert_ROI_scale500')

    mris_convertLH = pe.Node(interface=fs.MRIsConvert(), name='mris_convertLH')
    mris_convertLH.inputs.out_datatype = 'gii'
    mris_convertRH = mris_convertLH.clone('mris_convertRH')
    mris_convertRHwhite = mris_convertLH.clone('mris_convertRHwhite')
    mris_convertLHwhite = mris_convertLH.clone('mris_convertLHwhite')
    mris_convertRHinflated = mris_convertLH.clone('mris_convertRHinflated')
    mris_convertLHinflated = mris_convertLH.clone('mris_convertLHinflated')
    mris_convertLHlabels = mris_convertLH.clone('mris_convertLHlabels')
    mris_convertRHlabels = mris_convertLH.clone('mris_convertRHlabels')

    """
    Parcellation is performed given the aparc+aseg image from Freesurfer.
    The CMTK Parcellation step subdivides these regions to return a higher-resolution parcellation scheme.
    The parcellation used here is entitled "scale500" and returns 1015 regions.
    """

    parcellate = pe.Node(interface=cmtk.Parcellate(), name="Parcellate")
    parcellate.inputs.parcellation_name = parcellation_name

    """
    The CreateMatrix interface takes in the remapped aparc+aseg image as well as the label dictionary and fiber tracts
    and outputs a number of different files. The most important of which is the connectivity network itself, which is stored
    as a 'gpickle' and can be loaded using Python's NetworkX package (see CreateMatrix docstring). Also outputted are various
    NumPy arrays containing detailed tract information, such as the start and endpoint regions, and statistics on the mean and
    standard deviation for the fiber length of each connection. These matrices can be used in the ConnectomeViewer to plot the
    specific tracts that connect between user-selected regions.

    Here we choose the Lausanne2008 parcellation scheme, since we are incorporating the CMTK parcellation step.
    """

    creatematrix = pe.Node(interface=cmtk.CreateMatrix(), name="CreateMatrix")
    creatematrix.inputs.count_region_intersections = True

    """
    Next we define the endpoint of this tutorial, which is the CFFConverter node, as well as a few nodes which use
    the Nipype Merge utility. These are useful for passing lists of the files we want packaged in our CFF file.
    The inspect.getfile command is used to package this script into the resulting CFF file, so that it is easy to
    look back at the processing parameters that were used.
    """

    CFFConverter = pe.Node(interface=cmtk.CFFConverter(), name="CFFConverter")
    CFFConverter.inputs.script_files = op.abspath(inspect.getfile(inspect.currentframe()))
    giftiSurfaces = pe.Node(interface=util.Merge(6), name="GiftiSurfaces")
    giftiLabels = pe.Node(interface=util.Merge(2), name="GiftiLabels")
    niftiVolumes = pe.Node(interface=util.Merge(3), name="NiftiVolumes")
    fiberDataArrays = pe.Node(interface=util.Merge(4), name="FiberDataArrays")

    """
    We also create a node to calculate several network metrics on our resulting file, and another CFF converter
    which will be used to package these networks into a single file.
    """

    networkx = create_networkx_pipeline(name='networkx')
    cmats_to_csv = create_cmats_to_csv_pipeline(name='cmats_to_csv')
    nfibs_to_csv = pe.Node(interface=misc.Matlab2CSV(), name='nfibs_to_csv')
    merge_nfib_csvs = pe.Node(interface=misc.MergeCSVFiles(), name='merge_nfib_csvs')
    merge_nfib_csvs.inputs.extra_column_heading = 'Subject'
    merge_nfib_csvs.inputs.out_file = 'fibers.csv'
    NxStatsCFFConverter = pe.Node(interface=cmtk.CFFConverter(), name="NxStatsCFFConverter")
    NxStatsCFFConverter.inputs.script_files = op.abspath(inspect.getfile(inspect.currentframe()))

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    workflow.connect(
        [(inputnode, dtiproc, [("subjects_dir", "inputnode.subjects_dir"),
                               ("subject_id", "inputnode.subject_id"),
                               ("dwi", "inputnode.dwi"),
                               ("bvecs", "inputnode.bvecs"),
                               ("bvals", "inputnode.bvals")])
         ])

    workflow.connect([(inputnode, reg_label, [("subject_id", "inputnode.subject_id")])])
    #workflow.connect([(mri_convert_ROI_scale500, reg_label, [("out_file", "inputnode.manual_seg_rois")])])
    workflow.connect([(dtiproc, reg_label, [("outputnode.aparc_aseg", "inputnode.manual_seg_rois")])])
    
    workflow.connect([(dtiproc, reg_label, [("outputnode.wm_mask", "inputnode.wm_mask"),
                                            ("outputnode.term_mask", "inputnode.termination_mask"),
                                            ("outputnode.fa", "inputnode.fa"),
                                            ("outputnode.aparc_aseg", "inputnode.aparc_aseg"),
                      ])])

    workflow.connect([(reg_label, t1_to_dwi, [("outputnode.t1_to_dwi_matrix", "in_matrix_file")])])
    workflow.connect([(dtiproc, t1_to_dwi, [("outputnode.t1", "in_file")])])
    workflow.connect([(dtiproc, t1_to_dwi, [("outputnode.fa", "reference")])])
    workflow.connect([(inputnode, t1_to_dwi, [(('subject_id', add_subj_name_to_T1_dwi), 'out_file')])])

    workflow.connect([(reg_label, termmask_to_dwi, [("outputnode.t1_to_dwi_matrix", "in_matrix_file")])])
    workflow.connect([(dtiproc, termmask_to_dwi, [("outputnode.term_mask", "in_file")])])
    workflow.connect([(dtiproc, termmask_to_dwi, [("outputnode.fa", "reference")])])

    '''
    Connect outputnode
    '''

    workflow.connect([(t1_to_dwi, outputnode, [("out_file", "t1_to_dwi")])])

    workflow.connect([(dtiproc, outputnode, [("outputnode.t1", "t1"),
                                           ("outputnode.wm_prob", "wm_prob"),
                                           ("outputnode.gm_prob", "gm_prob"),
                                           ("outputnode.csf_prob", "csf_prob"),
                                           ("outputnode.single_fiber_mask", "single_fiber_mask"),
                                           ("outputnode.fa", "fa"),
                                           ("outputnode.rgb_fa", "rgb_fa"),
                                           ("outputnode.md", "md"),
                                           ("outputnode.mode", "mode"),
                                           ("outputnode.t1_brain", "t1_brain"),
                                           ("outputnode.wm_mask", "wm_mask"),
                                           ("outputnode.term_mask", "term_mask"),
                                           ("outputnode.aparc_aseg", "aparc_aseg"),
                                           ("outputnode.tissue_class_files", "tissue_class_files"),
                                           ])])

    
    workflow.connect([(reg_label, outputnode, [("outputnode.rois_to_dwi", "rois_to_dwi"),
                                           ("outputnode.wmmask_to_dwi", "wmmask_to_dwi"),
                                           ("outputnode.termmask_to_dwi", "termmask_to_dwi"),
                                           ("outputnode.dwi_to_t1_matrix", "dwi_to_t1_matrix"),
                                           ("outputnode.highres_t1_to_dwi_matrix", "highres_t1_to_dwi_matrix"),
                                           ])])

    workflow.connect([(dtiproc, outputnode, [("outputnode.aparc_aseg", "rois")])])

    tracking = anatomically_constrained_tracking("tracking")

    workflow.connect(
       [(inputnode, tracking, [("subject_id", "inputnode.subject_id"),
                               ("dwi", "inputnode.dwi"),
                               ("bvecs", "inputnode.bvecs"),
                               ("bvals", "inputnode.bvals"),
                               ])
         ])

    workflow.connect(
       [(reg_label, tracking, [("outputnode.wmmask_to_dwi", "inputnode.wm_mask"),
                               ("outputnode.termmask_to_dwi", "inputnode.termination_mask"),
                               ("outputnode.dwi_to_t1_matrix", "inputnode.registration_matrix_file"),
                               ])
         ])

    workflow.connect([(dtiproc, tracking, [("outputnode.t1", "inputnode.registration_image_file")])])

    workflow.connect([(dtiproc, tracking, [("outputnode.single_fiber_mask", "inputnode.single_fiber_mask")])])

    workflow.connect([(tracking, outputnode, [("outputnode.fiber_odfs", "fiber_odfs"),
                                           ("outputnode.fiber_tracks_tck_dwi", "fiber_tracks_tck_dwi"),
                                           ("outputnode.fiber_tracks_trk_t1", "fiber_tracks_trk_t1"),
                                           ])])


    workflow.connect([(tracking, creatematrix,[("outputnode.fiber_tracks_trk_t1","tract_file")])])

    workflow.connect([(inputnode, FreeSurferSource,[("subjects_dir","subjects_dir")])])
    workflow.connect([(inputnode, FreeSurferSource,[("subject_id","subject_id")])])

    workflow.connect([(inputnode, FreeSurferSourceLH,[("subjects_dir","subjects_dir")])])
    workflow.connect([(inputnode, FreeSurferSourceLH,[("subject_id","subject_id")])])

    workflow.connect([(inputnode, FreeSurferSourceRH,[("subjects_dir","subjects_dir")])])
    workflow.connect([(inputnode, FreeSurferSourceRH,[("subject_id","subject_id")])])

    workflow.connect([(inputnode, parcellate,[("subjects_dir","subjects_dir")])])
    workflow.connect([(inputnode, parcellate,[("subject_id","subject_id")])])
    workflow.connect([(parcellate, mri_convert_ROI_scale500,[('roi_file','in_file')])])

    """
    Surface conversions to GIFTI (pial, white, inflated, and sphere for both hemispheres)
    """

    workflow.connect([(FreeSurferSourceLH, mris_convertLH,[('pial','in_file')])])
    workflow.connect([(FreeSurferSourceRH, mris_convertRH,[('pial','in_file')])])
    workflow.connect([(FreeSurferSourceLH, mris_convertLHwhite,[('white','in_file')])])
    workflow.connect([(FreeSurferSourceRH, mris_convertRHwhite,[('white','in_file')])])
    workflow.connect([(FreeSurferSourceLH, mris_convertLHinflated,[('inflated','in_file')])])
    workflow.connect([(FreeSurferSourceRH, mris_convertRHinflated,[('inflated','in_file')])])

    """
    The annotation files are converted using the pial surface as a map via the MRIsConvert interface.
    One of the functions defined earlier is used to select the lh.aparc.annot and rh.aparc.annot files
    specifically (rather than e.g. rh.aparc.a2009s.annot) from the output list given by the FreeSurferSource.
    """

    workflow.connect([(FreeSurferSourceLH, mris_convertLHlabels,[('pial','in_file')])])
    workflow.connect([(FreeSurferSourceRH, mris_convertRHlabels,[('pial','in_file')])])
    workflow.connect([(FreeSurferSourceLH, mris_convertLHlabels, [(('annot', select_aparc_annot), 'annot_file')])])
    workflow.connect([(FreeSurferSourceRH, mris_convertRHlabels, [(('annot', select_aparc_annot), 'annot_file')])])


    workflow.connect(inputnode, 'resolution_network_file', creatematrix, 'resolution_network_file')
    workflow.connect([(inputnode, creatematrix,[("subject_id","out_matrix_file")])])
    workflow.connect([(inputnode, creatematrix,[("subject_id","out_matrix_mat_file")])])
    workflow.connect([(parcellate, creatematrix,[("roi_file","roi_file")])])

    workflow.connect([(creatematrix, fiberDataArrays,[("endpoint_file","in1")])])
    workflow.connect([(creatematrix, fiberDataArrays,[("endpoint_file_mm","in2")])])
    workflow.connect([(creatematrix, fiberDataArrays,[("fiber_length_file","in3")])])
    workflow.connect([(creatematrix, fiberDataArrays,[("fiber_label_file","in4")])])


    workflow.connect([(mris_convertLH, giftiSurfaces,[("converted","in1")])])
    workflow.connect([(mris_convertRH, giftiSurfaces,[("converted","in2")])])
    workflow.connect([(mris_convertLHwhite, giftiSurfaces,[("converted","in3")])])
    workflow.connect([(mris_convertRHwhite, giftiSurfaces,[("converted","in4")])])
    workflow.connect([(mris_convertLHinflated, giftiSurfaces,[("converted","in5")])])
    workflow.connect([(mris_convertRHinflated, giftiSurfaces,[("converted","in6")])])

    workflow.connect([(mris_convertLHlabels, giftiLabels,[("converted","in1")])])
    workflow.connect([(mris_convertRHlabels, giftiLabels,[("converted","in2")])])

    workflow.connect([(giftiSurfaces, CFFConverter,[("out","gifti_surfaces")])])
    workflow.connect([(giftiLabels, CFFConverter,[("out","gifti_labels")])])
    workflow.connect([(creatematrix, CFFConverter,[("matrix_files","gpickled_networks")])])
    workflow.connect([(fiberDataArrays, CFFConverter,[("out","data_files")])])
    workflow.connect([(inputnode, CFFConverter,[("subject_id","title")])])
    workflow.connect([(inputnode, CFFConverter, [(('subject_id', add_subj_name_to_Connectome), 'out_file')])])

    """
    The graph theoretical metrics which have been generated are placed into another CFF file.
    """

    workflow.connect([(inputnode, networkx,[("subject_id","inputnode.extra_field")])])
    workflow.connect([(creatematrix, networkx,[("intersection_matrix_file","inputnode.network_file")])])

    workflow.connect([(networkx, NxStatsCFFConverter,[("outputnode.network_files","gpickled_networks")])])
    workflow.connect([(giftiSurfaces, NxStatsCFFConverter,[("out","gifti_surfaces")])])
    workflow.connect([(giftiLabels, NxStatsCFFConverter,[("out","gifti_labels")])])
    workflow.connect([(fiberDataArrays, NxStatsCFFConverter,[("out","data_files")])])
    workflow.connect([(inputnode, NxStatsCFFConverter,[("subject_id","title")])])
    workflow.connect([(inputnode, NxStatsCFFConverter, [(('subject_id', add_subj_name_to_nxConnectome), 'out_file')])])

    workflow.connect([(CFFConverter, outputnode,[("connectome_file","connectome")])])
    workflow.connect([(NxStatsCFFConverter, outputnode,[("connectome_file","nxstatscff")])])
    workflow.connect([(creatematrix, outputnode,[("intersection_matrix_file","matrix_file")])])
    workflow.connect([(creatematrix, outputnode,[("matrix_mat_file","cmatrix")])])
    workflow.connect([(networkx, outputnode,[("outputnode.csv_files","nxcsv")])])
    return workflow
