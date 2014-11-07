import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs    # freesurfer
import nipype.interfaces.mrtrix as mrtrix
from nipype.workflows.misc.utils import select_aparc

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from coma.interfaces.dti import nonlinfit_fn
from coma.helpers import (add_subj_name_to_sfmask, add_subj_name_to_wmmask,
                          add_subj_name_to_T1, add_subj_name_to_T1brain,
                          add_subj_name_to_termmask, add_subj_name_to_aparc,
                          select_ribbon, select_GM, select_WM, select_CSF,
                          wm_labels_only)


def damaged_brain_dti_processing(name="dwi_preproc", use_FAST_masks=True):
    '''
    Uses both Freesurfer and FAST to mask the white matter
    because neither works sufficiently well in patients
    '''

    '''
    Define inputs and outputs of the workflow
    '''
    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=["subjects_dir",
                                                 "subject_id",
                                                 "dwi",
                                                 "bvecs",
                                                 "bvals"]),
        name="inputnode")

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=["single_fiber_mask",
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
                                                 "csf_prob"]),
        name="outputnode")

    '''
    Define the nodes
    '''
    nonlinfit_interface = util.Function(
        input_names=["dwi", "bvecs", "bvals", "base_name"],
        output_names=["tensor", "FA", "MD", "evecs", "evals",
                      "rgb_fa", "norm", "mode", "binary_mask", "b0_masked"],
        function=nonlinfit_fn)

    nonlinfit_node = pe.Node(
        interface=nonlinfit_interface, name="nonlinfit_node")

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


    fast_seg_T1 = pe.Node(interface=fsl.FAST(), name='fast_seg_T1')
    fast_seg_T1.inputs.segments = True
    fast_seg_T1.inputs.probability_maps = True

    if not use_FAST_masks:
        fix_wm_mask = pe.Node(interface=fsl.MultiImageMaths(), name='fix_wm_mask')
        fix_wm_mask.inputs.op_string = "-mul %s"

    if use_FAST_masks:
        make_termination_mask = pe.Node(
            interface=fsl.MultiImageMaths(), name='make_termination_mask')
        make_termination_mask.inputs.op_string = "-add %s -bin"
    else:
        make_termination_mask = pe.Node(
            interface=fsl.ImageMaths(), name='make_termination_mask')
        make_termination_mask.inputs.op_string = "-bin"

        fix_termination_mask = pe.Node(
            interface=fsl.MultiImageMaths(), name='fix_termination_mask')
        fix_termination_mask.inputs.op_string = "-binv -mul %s"


    wm_mask_interface = util.Function(input_names=["in_file", "out_filename"],
                                      output_names=["out_file"],
                                      function=wm_labels_only)

    make_wm_mask = pe.Node(interface=wm_mask_interface, name='make_wm_mask')

    MRmultiply = pe.Node(interface=mrtrix.MRMultiply(), name='MRmultiply')
    MRmultiply.inputs.out_filename = "Eroded_FA.nii.gz"

    MultFAbyMode = pe.Node(interface=mrtrix.MRMultiply(), name='MultFAbyMode')

    MRmult_merge = pe.Node(interface=util.Merge(2), name='MRmultiply_merge')

    MultFAbyMode_merge = pe.Node(
        interface=util.Merge(2), name='MultFAbyMode_merge')

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

    '''
    Connect the workflow
    '''
    workflow = pe.Workflow(name=name)
    workflow.base_dir = name

    '''
    Structural processing to create seed and termination masks
    '''
    workflow.connect(
        [(inputnode, FreeSurferSource, [("subjects_dir", "subjects_dir")])])
    workflow.connect(
        [(inputnode, FreeSurferSource, [("subject_id", "subject_id")])])
    workflow.connect(
        [(FreeSurferSource, mri_convert_T1, [('T1', 'in_file')])])
    workflow.connect(
        [(FreeSurferSource, mri_convert_Brain, [('brain', 'in_file')])])

    workflow.connect(
        [(inputnode, mri_convert_T1, [(('subject_id', add_subj_name_to_T1), 'out_file')])])
    workflow.connect(
        [(inputnode, mri_convert_Brain, [(('subject_id', add_subj_name_to_T1brain), 'out_file')])])
    workflow.connect(
        [(mri_convert_Brain, fast_seg_T1, [('out_file', 'in_files')])])
    workflow.connect(
        [(inputnode, fast_seg_T1, [("subject_id", "out_basename")])])

    workflow.connect(
        [(FreeSurferSource, mri_convert_ROIs, [(('aparc_aseg', select_aparc), 'in_file')])])
    workflow.connect(
        [(inputnode, mri_convert_ROIs, [(('subject_id', add_subj_name_to_aparc), 'out_file')])])

    if use_FAST_masks:
        workflow.connect(
            [(fast_seg_T1, outputnode, [(('tissue_class_files', select_WM), 'wm_mask')])])
        workflow.connect(
            [(fast_seg_T1, make_termination_mask, [(('tissue_class_files', select_GM), 'in_file')])])
        workflow.connect(
            [(fast_seg_T1, make_termination_mask, [(('tissue_class_files', select_WM), 'operand_files')])])
        workflow.connect(
            [(inputnode, make_termination_mask, [(('subject_id', add_subj_name_to_termmask), 'out_file')])])
        workflow.connect(
            [(make_termination_mask, outputnode, [("out_file", "term_mask")])])

    else:
        workflow.connect(
            [(mri_convert_ROIs, make_wm_mask, [('out_file', 'in_file')])])
        workflow.connect(
            [(make_wm_mask, fix_wm_mask, [('out_file', 'operand_files')])])
        workflow.connect(
            [(fast_seg_T1, fix_wm_mask, [(('tissue_class_files', select_WM), 'in_file')])])
        workflow.connect(
            [(FreeSurferSource, mri_convert_Ribbon, [(('ribbon', select_ribbon), 'in_file')])])
        workflow.connect(
            [(mri_convert_Ribbon, make_termination_mask, [('out_file', 'in_file')])])
        workflow.connect(
            [(make_termination_mask, fix_termination_mask, [('out_file', 'operand_files')])])
        workflow.connect(
            [(fast_seg_T1, fix_termination_mask, [(('tissue_class_files', select_CSF), 'in_file')])])

    '''
    Diffusion processing
    '''
    workflow.connect(inputnode, 'subject_id', nonlinfit_node, 'base_name')
    workflow.connect(inputnode, 'dwi', nonlinfit_node, 'dwi')
    workflow.connect(inputnode, 'bvecs', nonlinfit_node, 'bvecs')
    workflow.connect(inputnode, 'bvals', nonlinfit_node, 'bvals')

    workflow.connect(
        [(nonlinfit_node, median3d, [("binary_mask", "in_file")])])
    workflow.connect(
        [(median3d, erode_mask_firstpass, [("out_file", "in_file")])])
    workflow.connect(
        [(erode_mask_firstpass, erode_mask_secondpass, [("out_file", "in_file")])])
    workflow.connect([(nonlinfit_node, MRmult_merge, [("FA", "in1")])])
    workflow.connect(
        [(erode_mask_secondpass, MRmult_merge, [("out_file", "in2")])])
    workflow.connect([(MRmult_merge, MRmultiply, [("out", "in_files")])])
    workflow.connect([(MRmultiply, threshold_FA, [("out_file", "in_file")])])

    '''
    Create a single fiber mask
    '''
    workflow.connect([(nonlinfit_node, threshold_mode, [("mode", "in_file")])])
    workflow.connect(
        [(threshold_mode, MultFAbyMode_merge, [("out_file", "in1")])])
    workflow.connect(
        [(threshold_FA, MultFAbyMode_merge, [("out_file", "in2")])])
    workflow.connect(
        [(MultFAbyMode_merge, MultFAbyMode, [("out", "in_files")])])

    '''
    Fix output names with subject ID
    '''
    workflow.connect(
        [(inputnode, MultFAbyMode, [(('subject_id', add_subj_name_to_sfmask), 'out_filename')])])
    if not use_FAST_masks:
        workflow.connect(
            [(inputnode, make_wm_mask, [(('subject_id', add_subj_name_to_wmmask), 'out_filename')])])
        workflow.connect(
            [(inputnode, fix_wm_mask, [(('subject_id', add_subj_name_to_wmmask), 'out_file')])])
        workflow.connect(
            [(inputnode, fix_termination_mask, [(('subject_id', add_subj_name_to_termmask), 'out_file')])])


    '''
    Connect outputnode
    '''
    workflow.connect(
        [(fast_seg_T1, outputnode, [("tissue_class_files", "tissue_class_files")])])

    workflow.connect(
        [(fast_seg_T1, outputnode, [(('probability_maps', select_GM), 'gm_prob')])])
    workflow.connect(
        [(fast_seg_T1, outputnode, [(('probability_maps', select_WM), 'wm_prob')])])
    workflow.connect(
        [(fast_seg_T1, outputnode, [(('probability_maps', select_CSF), 'csf_prob')])])


    workflow.connect([
        (mri_convert_ROIs, outputnode, [("out_file", "aparc_aseg")]),
        (nonlinfit_node, outputnode, [("FA", "fa")]),
        (nonlinfit_node, outputnode, [("rgb_fa", "rgb_fa")]),
        (nonlinfit_node, outputnode, [("MD", "md")]),
        (nonlinfit_node, outputnode, [("mode", "mode")]),
        (MultFAbyMode, outputnode, [("out_file", "single_fiber_mask")]),
        (mri_convert_Brain, outputnode, [("out_file", "t1_brain")]),
        (mri_convert_T1, outputnode, [("out_file", "t1")]),
    ])

    if not use_FAST_masks:
        workflow.connect(
            [(fix_wm_mask, outputnode, [("out_file", "wm_mask")])])
        workflow.connect(
            [(fix_termination_mask, outputnode, [("out_file", "term_mask")])])

    return workflow
