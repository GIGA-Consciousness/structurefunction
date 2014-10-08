import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from coma.workflows.dti.basic import damaged_brain_dti_processing
from coma.workflows.dti.tracking import anatomically_constrained_tracking
from coma.workflows.dmn import create_paired_tract_analysis_wf
from coma.workflows.pet import create_pet_quantification_wf
from coma.labels import dmn_labels_combined
from coma.helpers import (add_subj_name_to_rois, rewrite_mat_for_applyxfm,
    add_subj_name_to_cortex_sfmask, add_subj_name_to_T1_dwi, select_GM)
from coma.interfaces.glucose import CMR_glucose, calculate_SUV, scale_PVC_matrix_fn

def coreg_without_resample(name="highres_coreg"):
    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=["fixed_image",
                                                 "moving_image",
                                                 "interp"]),
        name="inputnode")

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=["out_file",
                                                 "lowres_matrix_file",
                                                 "highres_matrix_file",
                                                 "resampled_fixed_image"]),
        name="outputnode")
    coregister_moving_to_fixed = pe.Node(interface=fsl.FLIRT(dof=12),
        name = 'coregister_moving_to_fixed')
    resample_fixed_to_moving = pe.Node(interface=fs.MRIConvert(),
        name = 'resample_fixed_to_moving')

    rewrite_mat_interface = util.Function(input_names=["in_matrix", "orig_img", "target_img", "shape", "vox_size"],
                                  output_names=["out_image", "out_matrix_file"],
                                  function=rewrite_mat_for_applyxfm)
    fix_FOV_in_matrix = pe.Node(interface=rewrite_mat_interface, name='fix_FOV_in_matrix')

    apply_fixed_matrix = pe.Node(interface=fsl.ApplyXfm(),
        name = 'apply_fixed_matrix')

    final_rigid_reg_to_fixed = pe.Node(interface=fsl.FLIRT(dof=6),
        name = 'final_rigid_reg_to_fixed')

    create_highres_xfm = pe.Node(interface=fsl.ConvertXFM(),
        name = 'create_highres_xfm')
    create_highres_xfm.inputs.concat_xfm = True

    workflow = pe.Workflow(name=name)

    workflow.connect([(inputnode, coregister_moving_to_fixed,[("moving_image","in_file")])])
    workflow.connect([(inputnode, coregister_moving_to_fixed,[("fixed_image","reference")])])
    workflow.connect([(coregister_moving_to_fixed, fix_FOV_in_matrix,[("out_matrix_file","in_matrix")])])
    workflow.connect([(inputnode, fix_FOV_in_matrix,[("moving_image","orig_img")])])
    workflow.connect([(inputnode, fix_FOV_in_matrix,[("fixed_image","target_img")])])

    workflow.connect([(inputnode, apply_fixed_matrix,[("moving_image","in_file")])])
    workflow.connect([(inputnode, apply_fixed_matrix,[("interp","interp")])])
    workflow.connect([(fix_FOV_in_matrix, apply_fixed_matrix,[("out_matrix_file","in_matrix_file")])])
    workflow.connect([(fix_FOV_in_matrix, apply_fixed_matrix,[("out_image","reference")])])

    workflow.connect([(inputnode, resample_fixed_to_moving,[('fixed_image','in_file')])])
    workflow.connect([(inputnode, resample_fixed_to_moving,[('moving_image','reslice_like')])])
    workflow.connect([(resample_fixed_to_moving, final_rigid_reg_to_fixed,[('out_file','reference')])])
    workflow.connect([(apply_fixed_matrix, final_rigid_reg_to_fixed,[('out_file','in_file')])])
    workflow.connect([(inputnode, final_rigid_reg_to_fixed,[('interp','interp')])])

    workflow.connect([(final_rigid_reg_to_fixed, create_highres_xfm,[('out_matrix_file','in_file2')])])
    workflow.connect([(fix_FOV_in_matrix, create_highres_xfm,[('out_matrix_file','in_file')])])

    workflow.connect([(coregister_moving_to_fixed, outputnode,[('out_matrix_file','lowres_matrix_file')])])
    workflow.connect([(create_highres_xfm, outputnode,[('out_file','highres_matrix_file')])])

    workflow.connect([(resample_fixed_to_moving, outputnode,[('out_file','resampled_fixed_image')])])
    workflow.connect([(final_rigid_reg_to_fixed, outputnode,[('out_file','out_file')])])
    return workflow

def create_reg_and_label_wf(name="reg_wf", manual_seg_rois=False):
    inputfields = ["subject_id",
                 "aparc_aseg",
                 "fa",
                 "wm_mask",
                 "termination_mask"]

    if manual_seg_rois:
        inputfields.append("manual_seg_rois")

    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=inputfields),
        name="inputnode")

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=["dwi_to_t1_matrix",
                                                 "t1_to_dwi_matrix",
                                                 "rois_to_dwi",
                                                 "rois",
                                                 "wmmask_to_dwi",
                                                 "termmask_to_dwi",
                                                 "highres_t1_to_dwi_matrix"]),
        name="outputnode")


    dmn_labels_if = util.Function(input_names=["in_file", "out_filename"],
                                  output_names=["out_file"], function=dmn_labels_combined)
    dmn_labelling = pe.Node(interface=dmn_labels_if, name='dmn_labelling')

    align_wmmask_to_dwi = coreg_without_resample("align_wmmask_to_fa")
    align_wmmask_to_dwi.inputs.inputnode.interp = "nearestneighbour"

    rois_to_dwi = pe.Node(interface=fsl.ApplyXfm(),
        name = 'rois_to_dwi')
    rois_to_dwi.inputs.interp = "nearestneighbour"

    threshold_fa = pe.Node(interface=fsl.ImageMaths(),
        name = 'threshold_fa')
    threshold_fa.inputs.op_string = "-thr 0.2 -bin"

    multiply_rois_by_termmask = pe.Node(interface=fsl.MultiImageMaths(),
        name = 'multiply_rois_by_termmask')
    multiply_rois_by_termmask.inputs.op_string = "-mul %s"

    termmask_to_dwi = rois_to_dwi.clone("termmask_to_dwi")

    invertxfm = pe.Node(interface=fsl.ConvertXFM(),
        name = 'invertxfm')
    invertxfm.inputs.invert_xfm = True


    '''
    Define renaming nodes
    '''
    rename_t1_to_dwi_mat = pe.Node(interface=util.Rename(format_string = "%(subject_id)s_t1_to_dwi_matrix"),
        name = 'rename_t1_to_dwi_mat')
    rename_t1_to_dwi_mat.inputs.keep_ext = True

    rename_dwi_to_t1_mat = pe.Node(interface=util.Rename(format_string = "%(subject_id)s_dwi_to_t1_matrix"),
        name = 'rename_dwi_to_t1_mat')
    rename_dwi_to_t1_mat.inputs.keep_ext = True

    rename_rois_dwi = pe.Node(interface=util.Rename(format_string = "%(subject_id)s_rois_dwi"),
        name = 'rename_rois_dwi')
    rename_rois_dwi.inputs.keep_ext = True

    rename_rois = pe.Node(interface=util.Rename(format_string = "%(subject_id)s_rois"),
        name = 'rename_rois')
    rename_rois.inputs.keep_ext = True

    rename_termmask_dwi = pe.Node(interface=util.Rename(format_string = "%(subject_id)s_term_mask_dwi"),
        name = 'rename_termmask_dwi')
    rename_termmask_dwi.inputs.keep_ext = True
    
    rename_wmmask_dwi = pe.Node(interface=util.Rename(format_string = "%(subject_id)s_wm_mask_dwi"),
        name = 'rename_wmmask_dwi')
    rename_wmmask_dwi.inputs.keep_ext = True

    rename_highres_matrix_file = pe.Node(interface=util.Rename(format_string = "%(subject_id)s_t1_to_dwi_NoResample"),
        name = 'rename_highres_matrix_file')
    rename_highres_matrix_file.inputs.keep_ext = True

    workflow = pe.Workflow(name=name)

    workflow.connect([(inputnode, align_wmmask_to_dwi,[("wm_mask","inputnode.moving_image")])])
    workflow.connect([(inputnode, threshold_fa,[("fa","in_file")])])
    workflow.connect([(threshold_fa, align_wmmask_to_dwi,[("out_file","inputnode.fixed_image")])])
    

    if manual_seg_rois:
        workflow.connect([(inputnode, rois_to_dwi,[("manual_seg_rois","in_file")])])
        workflow.connect([(inputnode, rois_to_dwi,[("manual_seg_rois","reference")])])
        workflow.connect([(inputnode, outputnode, [("manual_seg_rois", "rois")])])

    else:
        workflow.connect(
            [(inputnode, dmn_labelling, [(('subject_id', add_subj_name_to_rois), 'out_filename')])])
        workflow.connect([(inputnode, dmn_labelling, [("aparc_aseg", "in_file")])])

        workflow.connect([(dmn_labelling, multiply_rois_by_termmask,[("out_file","in_file")])])
        workflow.connect([(inputnode, multiply_rois_by_termmask,[("termination_mask","operand_files")])])
        workflow.connect([(multiply_rois_by_termmask, rename_rois,[("out_file","in_file")])])
        workflow.connect([(inputnode, rename_rois,[("subject_id","subject_id")])])
        workflow.connect([(rename_rois, rois_to_dwi,[("out_file","in_file")])])
        workflow.connect([(rename_rois, rois_to_dwi,[("out_file","reference")])])
        workflow.connect(
            [(rename_rois, outputnode, [("out_file", "rois")])])


    workflow.connect([(align_wmmask_to_dwi, rois_to_dwi,[("outputnode.highres_matrix_file","in_matrix_file")])])

    workflow.connect([(inputnode, termmask_to_dwi,[("termination_mask","in_file")])])
    workflow.connect([(inputnode, termmask_to_dwi,[("termination_mask","reference")])])
    workflow.connect([(align_wmmask_to_dwi, termmask_to_dwi,[("outputnode.highres_matrix_file","in_matrix_file")])])

    workflow.connect(
        [(align_wmmask_to_dwi, invertxfm, [("outputnode.lowres_matrix_file", "in_file")])])


    workflow.connect(
        [(inputnode, rename_t1_to_dwi_mat, [("subject_id", "subject_id")])])
    workflow.connect(
        [(align_wmmask_to_dwi, rename_t1_to_dwi_mat, [("outputnode.lowres_matrix_file", "in_file")])])
    workflow.connect(
        [(rename_t1_to_dwi_mat, outputnode, [("out_file", "t1_to_dwi_matrix")])])


    workflow.connect(
        [(inputnode, rename_dwi_to_t1_mat, [("subject_id", "subject_id")])])
    workflow.connect(
        [(invertxfm, rename_dwi_to_t1_mat, [("out_file", "in_file")])])
    workflow.connect(
        [(rename_dwi_to_t1_mat, outputnode, [("out_file", "dwi_to_t1_matrix")])])

    workflow.connect(
        [(inputnode, rename_rois_dwi, [("subject_id", "subject_id")])])
    workflow.connect(
        [(rois_to_dwi, rename_rois_dwi, [("out_file", "in_file")])])
    workflow.connect(
        [(rename_rois_dwi, outputnode, [("out_file", "rois_to_dwi")])])

    workflow.connect(
        [(inputnode, rename_termmask_dwi, [("subject_id", "subject_id")])])
    workflow.connect(
        [(termmask_to_dwi, rename_termmask_dwi, [("out_file", "in_file")])])
    workflow.connect(
        [(rename_termmask_dwi, outputnode, [("out_file", "termmask_to_dwi")])])

    workflow.connect(
        [(inputnode, rename_wmmask_dwi, [("subject_id", "subject_id")])])
    workflow.connect(
        [(align_wmmask_to_dwi, rename_wmmask_dwi, [("outputnode.out_file", "in_file")])])
    workflow.connect(
        [(rename_wmmask_dwi, outputnode, [("out_file", "wmmask_to_dwi")])])

    workflow.connect(
        [(inputnode, rename_highres_matrix_file, [("subject_id", "subject_id")])])
    workflow.connect(
        [(align_wmmask_to_dwi, rename_highres_matrix_file, [("outputnode.highres_matrix_file", "in_file")])])
    workflow.connect(
        [(rename_highres_matrix_file, outputnode, [("out_file", "highres_t1_to_dwi_matrix")])])
    return workflow


def create_dmn_pipeline_step1(name="dmn_step1", scale_by_glycemia=True, manual_seg_rois=False):
    inputfields = ["subjects_dir",
                     "subject_id",
                     "dwi",
                     "bvecs",
                     "bvals",
                     "fdgpet",
                     "dose",
                     "weight",
                     "delay",
                     "glycemie",
                     "scan_time"]

    if manual_seg_rois:
        inputfields.append("manual_seg_rois")

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
                                                 "rois",
                                                 "rois_to_dwi",
                                                 "wmmask_to_dwi",
                                                 "termmask_to_dwi",
                                                 "dwi_to_t1_matrix",
                                                 "highres_t1_to_dwi_matrix",

                                                 # Outputs from the PET workflow after SUV calculation
                                                 "SUV_corrected_pet_to_t1",
                                                 "AIF_corrected_pet_to_t1",
                                                 "pet_results_npz",
                                                 "pet_results_mat",
                                                 "orig_pet_to_t1",

                                                 # T1 in DWI space for reference
                                                 "t1_to_dwi",
                                                 "single_fiber_mask_cortex_only",
                                                 ]),
        name="outputnode")

    t1_to_dwi = pe.Node(interface=fsl.ApplyXfm(),
        name = 't1_to_dwi')

    termmask_to_dwi = t1_to_dwi.clone("termmask_to_dwi")

    compute_cmr_glc_interface = util.Function(input_names=["subject_id", "in_file", "dose", "weight", "delay",
        "glycemie", "scan_time"], output_names=["out_file", "cax2", "mecalc", "denom"], function=CMR_glucose)
    compute_AIF_PET = pe.Node(interface=compute_cmr_glc_interface, name='compute_AIF_PET')

    compute_SUV_interface = util.Function(input_names=["subject_id", "in_file", "dose", "weight", "delay",
        "scan_time", "isotope", 'height', "glycemie"],
        output_names=["out_file"], function=calculate_SUV)
    compute_SUV_norm_glycemia = pe.Node(interface=compute_SUV_interface, name='compute_SUV_norm_glycemia')

    scale_PVC_matrix_interface = util.Function(input_names=["subject_id", "in_file", "dose", "weight", "delay",
        "scan_time", "isotope", 'height', "glycemie", "scale_SUV_by_glycemia"],
        output_names=["out_npz", "out_matlab_mat"], function=scale_PVC_matrix_fn)
    scale_PVC_matrix = pe.Node(interface=scale_PVC_matrix_interface, name='scale_PVC_matrix')
    scale_PVC_matrix.inputs.scale_SUV_by_glycemia = scale_by_glycemia

    single_fiber_mask_cortex_only = pe.Node(
        interface=fsl.MultiImageMaths(), name='single_fiber_mask_cortex_only')
    single_fiber_mask_cortex_only.inputs.op_string = "-mul %s"

    dtiproc = damaged_brain_dti_processing("dtiproc")
    reg_label = create_reg_and_label_wf("reg_label", manual_seg_rois)
    petquant = create_pet_quantification_wf("petquant", segment_t1=False)

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

    if manual_seg_rois:
        workflow.connect([(inputnode, reg_label, [("manual_seg_rois", "inputnode.manual_seg_rois")])])
    
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

    workflow.connect([(inputnode, single_fiber_mask_cortex_only, [(('subject_id', add_subj_name_to_cortex_sfmask), 'out_file')])])
    workflow.connect([(termmask_to_dwi, single_fiber_mask_cortex_only, [("out_file", "operand_files")])])
    workflow.connect([(dtiproc, single_fiber_mask_cortex_only, [("outputnode.single_fiber_mask", "in_file")])])

    workflow.connect([(inputnode, compute_SUV_norm_glycemia, [("subject_id", "subject_id"),
                                                   ("dose", "dose"),
                                                   ("weight", "weight"),
                                                   ("delay", "delay"),
                                                   ("scan_time", "scan_time"),
                                                   ])])
    
    if scale_by_glycemia == True:
        workflow.connect([(inputnode, compute_SUV_norm_glycemia, [("glycemie", "glycemie")])])

    # This is for the arterial input function approximation for the FDG uptake
    workflow.connect([(inputnode, compute_AIF_PET, [("subject_id", "subject_id"),
                                                   ("dose", "dose"),
                                                   ("weight", "weight"),
                                                   ("delay", "delay"),
                                                   ("glycemie", "glycemie"),
                                                   ("scan_time", "scan_time"),
                                                   ])])

    workflow.connect([(inputnode, scale_PVC_matrix, [("subject_id", "subject_id"),
                                                   ("dose", "dose"),
                                                   ("weight", "weight"),
                                                   ("delay", "delay"),
                                                   ("glycemie", "glycemie"),
                                                   ("scan_time", "scan_time"),
                                                   ])])


    workflow.connect(
        [(dtiproc, petquant, [(('outputnode.tissue_class_files', select_GM), 'inputnode.gm_mask')])])


    workflow.connect([(dtiproc, petquant, [("outputnode.t1", "inputnode.t1"),
                                           ("outputnode.wm_prob", "inputnode.wm_prob"),
                                           ("outputnode.gm_prob", "inputnode.gm_prob"),
                                           ("outputnode.csf_prob", "inputnode.csf_prob"),
                                           ])])

    workflow.connect([(inputnode, petquant, [("fdgpet", "inputnode.pet")])])
    workflow.connect([(inputnode, petquant, [("subject_id", "inputnode.subject_id")])])

    if manual_seg_rois:
        workflow.connect([(inputnode, petquant, [("manual_seg_rois", "inputnode.rois")])])
    else:
        workflow.connect([(reg_label, petquant, [("outputnode.rois", "inputnode.rois")])])

    workflow.connect([(petquant, compute_AIF_PET, [("outputnode.corrected_pet_to_t1", "in_file")])])
    workflow.connect([(petquant, compute_SUV_norm_glycemia, [("outputnode.corrected_pet_to_t1", "in_file")])])
    workflow.connect([(petquant, scale_PVC_matrix, [("outputnode.pet_results_npz", "in_file")])])

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
    if manual_seg_rois:
        workflow.connect([(inputnode, outputnode, [("manual_seg_rois", "rois")])])
    else:
        workflow.connect([(reg_label, outputnode, [("outputnode.rois", "rois")])])

    workflow.connect([(compute_AIF_PET, outputnode, [("out_file", "SUV_corrected_pet_to_t1")])])
    workflow.connect([(compute_SUV_norm_glycemia, outputnode, [("out_file", "AIF_corrected_pet_to_t1")])])
    workflow.connect([(petquant, outputnode, [("outputnode.orig_pet_to_t1", "orig_pet_to_t1")])])
    workflow.connect([(scale_PVC_matrix, outputnode, [("out_npz", "pet_results_npz")])])
    workflow.connect([(scale_PVC_matrix, outputnode, [("out_matlab_mat", "pet_results_mat")])])
    workflow.connect([(single_fiber_mask_cortex_only, outputnode, [("out_file", "single_fiber_mask_cortex_only")])])
    return workflow


def create_dmn_pipeline_step2(name="dmn_step2", auto_reorient=True):
    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=[# For the fiber tracking
                                                 "subject_id",
                                                 "dwi",
                                                 "bvecs",
                                                 "bvals",
                                                 "single_fiber_mask",
                                                 "wm_mask",
                                                 "termination_mask",
                                                 "registration_matrix_file",
                                                 "registration_image_file",

                                                 # For the connectivity workflow
                                                 "fa",
                                                 "md",
                                                 "roi_file",
                                                 "summary_images"]),
        name="inputnode")

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=["fiber_odfs",
                                                 "fiber_tracks_tck_dwi",
                                                 "fiber_tracks_trk_t1",
                                                 "connectivity_files",
                                                 "connectivity_data",
                                                 "summary_images"]),
        name="outputnode")

    tracking = anatomically_constrained_tracking("tracking")
    connectivity = create_paired_tract_analysis_wf("connectivity")

    workflow = pe.Workflow(name=name)
    workflow.base_dir = name

    workflow.connect(
       [(inputnode, tracking, [("subject_id", "inputnode.subject_id"),
                               ("dwi", "inputnode.dwi"),
                               ("bvecs", "inputnode.bvecs"),
                               ("bvals", "inputnode.bvals"),
                               ("single_fiber_mask", "inputnode.single_fiber_mask"),
                               ("wm_mask", "inputnode.wm_mask"),
                               ("termination_mask", "inputnode.termination_mask"),
                               ("registration_matrix_file", "inputnode.registration_matrix_file"),
                               ("registration_image_file", "inputnode.registration_image_file"),
                               ])
         ])


    workflow.connect([(tracking, connectivity, [("outputnode.fiber_tracks_tck_dwi", "inputnode.track_file")])])

    workflow.connect(
        [(inputnode, connectivity, [("subject_id", "inputnode.subject_id"),
                                    ("fa", "inputnode.fa"),
                                    ("md", "inputnode.md"),
                                    ("roi_file", "inputnode.roi_file"),
                                    ("registration_matrix_file", "inputnode.registration_matrix_file"),
                                    ("registration_image_file", "inputnode.registration_image_file"),
                                    ])
         ])

    workflow.connect([(tracking, outputnode, [("outputnode.fiber_odfs", "fiber_odfs"),
                                           ("outputnode.fiber_tracks_tck_dwi", "fiber_tracks_tck_dwi"),
                                           ("outputnode.fiber_tracks_trk_t1", "fiber_tracks_trk_t1"),
                                           ])])

    workflow.connect([(connectivity, outputnode, [("outputnode.connectivity_files", "connectivity_files"),
                                           ("outputnode.connectivity_data", "connectivity_data"),
                                           ("outputnode.summary_images", "summary_images"),
                                           ])])

    return workflow
