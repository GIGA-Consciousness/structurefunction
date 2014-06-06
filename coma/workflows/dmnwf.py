import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from coma.workflows.dti.basic import damaged_brain_dti_processing
from coma.workflows.dmn import create_paired_tract_analysis_wf
from coma.workflows.pet import create_pet_quantification_wf
from coma.labels import dmn_labels_combined
from coma.helpers import add_subj_name_to_rois, rewrite_mat_for_applyxfm

def coreg_without_resample(name="highres_coreg"):
    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=["fixed_image",
                                                 "moving_image",
                                                 "interp",
                                                 "cost"]),
        name="inputnode")

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=["out_file",
                                                 "low_res_out_matrix_file",
                                                 #"high_res_out_matrix_file",
                                                 "resampled_fixed_image"]),
        name="outputnode")
    coregister_moving_to_fixed = pe.Node(interface=fsl.FLIRT(dof=12),
        name = 'coregister_moving_to_fixed')
    resample_fixed_to_moving = pe.Node(interface=fs.MRIConvert(),
        name = 'resample_fixed_to_moving')

    rewrite_mat_interface = util.Function(input_names=["in_matrix", "orig_img", "target_img", "shape", "vox_size"],
                                  output_names=["out_image", "out_matrix"],
                                  function=rewrite_mat_for_applyxfm)
    fix_FOV_in_matrix = pe.Node(interface=rewrite_mat_interface, name='fix_FOV_in_matrix')

    apply_fixed_matrix = pe.Node(interface=fsl.ApplyXfm(),
        name = 'apply_fixed_matrix')

    final_rigid_reg_to_fixed = pe.Node(interface=fsl.FLIRT(dof=6),
        name = 'final_rigid_reg_to_fixed')

    #out_image, out_matrix = rewrite_mat_for_applyxfm('WMtoFA.mat',
    #    orig_img="Bend1_wm_seed_mask.nii.gz", target_img="Bend1_fa.nii.gz", shape=[256,256,256], vox_size=[1,1,1])
    #os.system("flirt -in Bend1_wm_seed_mask.nii.gz -ref HeaderImage.nii.gz -applyxfm -init Transform.mat -out Final.nii")

    #os.system("flirt -in Final.nii.gz -ref fa_resamp.nii.gz -out FinalFix.nii")

    workflow = pe.Workflow(name=name)

    workflow.connect([(inputnode, coregister_moving_to_fixed,[("moving_image","in_file")])])
    workflow.connect([(inputnode, coregister_moving_to_fixed,[("fixed_image","reference")])])
    workflow.connect([(coregister_moving_to_fixed, fix_FOV_in_matrix,[("out_matrix_file","in_matrix")])])
    workflow.connect([(inputnode, fix_FOV_in_matrix,[("moving_image","orig_img")])])
    workflow.connect([(inputnode, fix_FOV_in_matrix,[("fixed_image","target_img")])])

    workflow.connect([(inputnode, apply_fixed_matrix,[("moving_image","in_file")])])
    workflow.connect([(fix_FOV_in_matrix, apply_fixed_matrix,[("out_matrix","in_matrix_file")])])
    workflow.connect([(fix_FOV_in_matrix, apply_fixed_matrix,[("out_image","reference")])])

    workflow.connect([(inputnode, resample_fixed_to_moving,[('fixed_image','in_file')])])
    workflow.connect([(inputnode, resample_fixed_to_moving,[('moving_image','reslice_like')])])
    workflow.connect([(resample_fixed_to_moving, final_rigid_reg_to_fixed,[('out_file','reference')])])
    workflow.connect([(apply_fixed_matrix, final_rigid_reg_to_fixed,[('out_file','in_file')])])

    workflow.connect([(final_rigid_reg_to_fixed, outputnode,[('out_file','out_file')])])
    return workflow

def create_reg_and_label_wf(name="reg_wf"):
    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=["subject_id",
                                                 "aparc_aseg",
                                                 "fa",
                                                 "wm_mask"]),
        name="inputnode")

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=["dwi_to_t1_matrix",
                                                 "t1_to_dwi_matrix",
                                                 "rois_to_dwi",
                                                 "rois"]),
        name="outputnode")

    coregister = pe.Node(interface=fsl.FLIRT(dof=12), name = 'coregister')
    coregister.inputs.cost = ('normmi')
    coregister.inputs.interp = 'trilinear'

    convertxfm = pe.Node(interface=fsl.ConvertXFM(), name='convertxfm')
    convertxfm.inputs.invert_xfm = True

    rois_to_dwispace = pe.Node(interface=fsl.ApplyXfm(), name='rois_to_dwispace')
    rois_to_dwispace.inputs.apply_xfm = True
    rois_to_dwispace.inputs.interp = 'nearestneighbour'

    invertxfm = pe.Node(interface=fsl.ConvertXFM(), name = 'invertxfm')
    invertxfm.inputs.invert_xfm = True

    coreg_resamp = pe.Node(interface=fsl.FLIRT(dof=12), name = 'coreg_resamp')
    coreg_resamp.inputs.cost = ('normmi')
    coreg_resamp.inputs.interp = 'trilinear'

    resample_fa = pe.Node(interface=fs.MRIConvert(), name = 'resample_fa')
    resample_fa.inputs.out_file = "fa_resamp.nii.gz"
    resample_fa.inputs.out_type = 'nii'

    dmn_labels_if = util.Function(input_names=["in_file", "out_filename"],
                                  output_names=["out_file"], function=dmn_labels_combined)
    dmn_labelling = pe.Node(interface=dmn_labels_if, name='dmn_labelling')

    workflow = pe.Workflow(name=name)
    workflow.connect(
        [(inputnode, dmn_labelling, [(('subject_id', add_subj_name_to_rois), 'out_filename')])])
    workflow.connect([(inputnode, dmn_labelling, [("aparc_aseg", "in_file")])])

    workflow.connect([(inputnode, coregister,[("wm_mask","in_file")])])
    workflow.connect([(inputnode, coregister,[('fa','reference')])])
    workflow.connect([(coregister, invertxfm,[("out_matrix_file","in_file")])])

    workflow.connect([(inputnode, coreg_resamp,[("wm_mask","in_file")])])
    workflow.connect([(resample_fa, coreg_resamp,[('out_file','reference')])])

    coreg_resamp = pe.Node(interface=fixmat_interface(dof=12), name = 'coreg_resamp')
    coreg_resamp.inputs.cost = ('normmi')
    coreg_resamp.inputs.interp = 'trilinear'

    coreg_resamp = pe.Node(interface=fsl.FLIRT(dof=12), name = 'coreg_resamp')
    coreg_resamp.inputs.cost = ('normmi')
    coreg_resamp.inputs.interp = 'trilinear'

    #os.system("flirt -in Bend1_wm_seed_mask.nii.gz -ref HeaderImage.nii.gz -applyxfm -init Transform.mat -out Final.nii")

    #os.system("flirt -in Final.nii.gz -ref fa_resamp.nii.gz -out FinalFix.nii")

    #workflow.connect([(dmn_labelling, rois_to_dwispace, [("out_file", "in_file")])])
    #workflow.connect([(resample_fa, rois_to_dwispace, [("out_file", "reference")])])

    workflow.connect([(dmn_labelling, rois_to_dwispace, [("out_file", "in_file")])])
    # Only to specify FOV and voxel size
    workflow.connect([(inputnode, resample_fa, [("fa", "in_file")])])
    workflow.connect([(dmn_labelling, resample_fa, [("out_file", "reslice_like")])])
    workflow.connect([(resample_fa, rois_to_dwispace, [("out_file", "reference")])])
    workflow.connect([(coreg_resamp, rois_to_dwispace, [('out_matrix_file','in_matrix_file')])])

    workflow.connect(
        [(dmn_labelling, outputnode, [("out_file", "rois")])])
    workflow.connect(
        [(coregister, outputnode, [("out_matrix_file", "t1_to_dwi_matrix")])])
    workflow.connect(
        [(invertxfm, outputnode, [("out_file", "dwi_to_t1_matrix")])])
    workflow.connect(
        [(rois_to_dwispace, outputnode, [("out_file", "rois_to_dwi")])])
    return workflow


def create_dmn_pipeline_step1(name="dmn_step1", auto_reorient=True):
    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=["subjects_dir",
                                                 "subject_id",
                                                 "dwi",
                                                 "bvecs",
                                                 "bvals",
                                                 "fdgpet"]),
        name="inputnode")

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=["fa",
                                                 "rgb_fa",
                                                 "md",
                                                 "csdeconv",
                                                 "tracts_tck", "rois", "t1",
                                                 "t1_brain", "wmmask_dtispace", "fa_t1space", "summary", "filtered_tractographies",
                                                 "fiber_label_file", "intersection_matrix_mat_file"]),
        name="outputnode")

    dtiproc = damaged_brain_dti_processing("dtiproc")
    reg_label = create_reg_and_label_wf("reg_label")
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
    
    workflow.connect([(dtiproc, reg_label, [("outputnode.wm_prob", "inputnode.wm_mask"),
                                            ("outputnode.fa", "inputnode.fa"),
                                            ("outputnode.aparc_aseg", "inputnode.aparc_aseg"),
                      ])])



    workflow.connect([(inputnode, petquant, [("fdgpet", "inputnode.pet")])])
    workflow.connect([(reg_label, petquant, [("outputnode.rois", "inputnode.rois")])])
    workflow.connect([(dtiproc, petquant, [("outputnode.t1", "inputnode.t1"),
                                           ("outputnode.wm_prob", "inputnode.wm_prob"),
                                           ("outputnode.gm_prob", "inputnode.gm_prob"),
                                           ("outputnode.csf_prob", "inputnode.csf_prob"),
                                           ])])

    '''
    Connect outputnode
    '''
    #workflow.connect([(petquant, petquant, [("rois", "inputnode.rois")])])
    # "out_files",
    #                                                  "pet_to_t1",
    #                                                  "corrected_pet_to_t1",
    #                                                  "pet_results_npz",
    #                                                  "pet_results_mat"
    # "single_fiber_mask",
    #                                                  "fa",
    #                                                  "rgb_fa",
    #                                                  "md",
    #                                                  "mode",
    #                                                  "t1",
    #                                                  "t1_brain",
    #                                                  "wm_mask",
    #                                                  "term_mask",
    #                                                  "aparc_aseg",
    #                                                  "tissue_class_files",
    #                                                  "gm_prob",
    #                                                  "wm_prob",
    #                                                  "csf_prob"
    # "dwi_to_t1_matrix",
    #                                                  "t1_to_dwi_matrix",
    #                                                  "rois_to_dwi",
    #                                                  "rois"
    return workflow


def create_dmn_pipeline_step2(name="dmn_step2", auto_reorient=True):
    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=["track_file",
                                                 "fa",
                                                 "md",
                                                 "roi_file"]),
        name="inputnode")

    # fiber tracking
    connectivity = create_paired_tract_analysis_wf("connectivity")

    workflow = pe.Workflow(name=name)
    workflow.base_dir = name

    workflow.connect(
        [(inputnode, connectivity, [("track_file", "inputnode.track_file"),
                                    ("fa", "inputnode.fa"),
                                    ("md", "inputnode.md"),
                                    ("roi_file", "inputnode.roi_file")])
         ])
    return workflow
