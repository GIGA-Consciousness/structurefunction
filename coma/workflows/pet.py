import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl
from ..helpers import select_CSF, select_WM, select_GM

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from coma.interfaces.pve import PartialVolumeCorrection


def create_pet_quantification_wf(name="petquant", segment_t1=True):

    '''
    Define inputs and outputs of the workflow
    '''
    if segment_t1:
        inputnode = pe.Node(
            interface=util.IdentityInterface(fields=["subject_id",
                                                     "t1",
                                                     "rois",
                                                     "pet"]),
            name="inputnode")
    else:
        inputnode = pe.Node(
            interface=util.IdentityInterface(fields=["subject_id",
                                                     "t1",
                                                     "gm_prob",
                                                     "wm_prob",
                                                     "csf_prob",
                                                     "rois",
                                                     "pet"]),
            name="inputnode")

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=["out_files",
                                                 "pet_to_t1",
                                                 "corrected_pet_to_t1",
                                                 "pet_results_npz",
                                                 "pet_results_mat"]),
        name="outputnode")

    '''
    Define the nodes
    '''
    if segment_t1:
        fast_seg_T1 = pe.Node(interface=fsl.FAST(), name='fast_seg_T1')
        fast_seg_T1.inputs.segments = True
        fast_seg_T1.inputs.probability_maps = True

    coregister = pe.Node(interface=fsl.FLIRT(dof=6), name='coregister')
    coregister.inputs.cost = ('corratio')
    coregister.inputs.interp = 'trilinear'

    convertxfm = pe.Node(interface=fsl.ConvertXFM(), name='convertxfm')
    convertxfm.inputs.invert_xfm = True

    applyxfm_t1 = pe.Node(interface=fsl.ApplyXfm(), name='applyxfm_t1')
    applyxfm_t1.inputs.apply_xfm = True
    applyxfm_t1.inputs.interp = 'trilinear'

    applyxfm_gm = applyxfm_t1.clone("applyxfm_gm")
    applyxfm_gm.inputs.interp = 'nearestneighbour'
    applyxfm_wm = applyxfm_gm.clone("applyxfm_wm")
    applyxfm_csf = applyxfm_gm.clone("applyxfm_csf")

    applyxfm_rois = applyxfm_t1.clone("applyxfm_rois")
    applyxfm_rois.inputs.interp = 'nearestneighbour'

    pve_correction = pe.Node(
        interface=PartialVolumeCorrection(), name='pve_correction')
    pve_correction.inputs.skip_atlas = False
    pve_correction.inputs.use_fs_LUT = False

    applyxfm_CorrectedPET = pe.Node(interface=fsl.ApplyXfm(), name = 'applyxfm_CorrectedPET')
    applyxfm_CorrectedPET.inputs.apply_xfm = True
    applyxfm_CorrectedPET.inputs.interp = 'trilinear'

    '''
    Connect the workflow
    '''
    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    workflow.connect(
        [(inputnode, coregister, [('t1', 'reference')])])
    workflow.connect(
        [(inputnode, coregister, [('pet', 'in_file')])])
    workflow.connect(
        [(coregister, convertxfm, [('out_matrix_file', 'in_file')])])

    if segment_t1:
        workflow.connect(
            [(inputnode, fast_seg_T1, [('t1', 'in_files')])])
        workflow.connect(
            [(inputnode, fast_seg_T1, [('subject_id', 'out_basename')])])

    workflow.connect(
        [(convertxfm, applyxfm_t1, [('out_file', 'in_matrix_file')])])
    workflow.connect(
        [(convertxfm, applyxfm_gm, [('out_file', 'in_matrix_file')])])
    workflow.connect(
        [(convertxfm, applyxfm_wm, [('out_file', 'in_matrix_file')])])
    workflow.connect(
        [(convertxfm, applyxfm_csf, [('out_file', 'in_matrix_file')])])
    workflow.connect(
        [(convertxfm, applyxfm_rois, [('out_file', 'in_matrix_file')])])

    workflow.connect(
        [(inputnode, applyxfm_t1, [('pet', 'reference')])])
    workflow.connect(
        [(inputnode, applyxfm_t1, [('t1', 'in_file')])])

    workflow.connect(
        [(inputnode, applyxfm_gm, [('pet', 'reference')])])
    workflow.connect(
        [(inputnode, applyxfm_wm, [('pet', 'reference')])])
    workflow.connect(
        [(inputnode, applyxfm_csf, [('pet', 'reference')])])


    if segment_t1:
        workflow.connect(
            [(fast_seg_T1, applyxfm_gm, [(('probability_maps', select_GM), 'in_file')])])
        workflow.connect(
            [(fast_seg_T1, applyxfm_wm, [(('probability_maps', select_WM), 'in_file')])])
        workflow.connect(
            [(fast_seg_T1, applyxfm_csf, [(('probability_maps', select_CSF), 'in_file')])])
    else:
        workflow.connect(
            [(inputnode, applyxfm_gm, [('gm_prob', 'in_file')])])
        workflow.connect(
            [(inputnode, applyxfm_wm, [('wm_prob', 'in_file')])])
        workflow.connect(
            [(inputnode, applyxfm_csf, [('csf_prob', 'in_file')])])

    workflow.connect(
        [(inputnode, applyxfm_rois, [('pet', 'reference')])])
    workflow.connect(
        [(inputnode, applyxfm_rois, [('rois', 'in_file')])])

    workflow.connect(
        [(inputnode, pve_correction, [('pet', 'pet_file')])])
    workflow.connect(
        [(applyxfm_t1, pve_correction, [('out_file', 't1_file')])])
    workflow.connect(
        [(applyxfm_gm, pve_correction, [('out_file', 'grey_matter_file')])])
    workflow.connect(
        [(applyxfm_wm, pve_correction, [('out_file', 'white_matter_file')])])
    workflow.connect(
        [(applyxfm_csf, pve_correction, [('out_file', 'csf_file')])])
    workflow.connect(
        [(applyxfm_rois, pve_correction, [('out_file', 'roi_file')])])
    workflow.connect(
        [(pve_correction, applyxfm_CorrectedPET, [('mueller_gartner_rousset', 'in_file')])])

    workflow.connect(
        [(inputnode, applyxfm_CorrectedPET, [('t1', 'reference')])])
    workflow.connect(
        [(coregister, applyxfm_CorrectedPET, [('out_matrix_file', 'in_matrix_file')])])

    '''
    Connect outputnode
    '''
    workflow.connect(
        [(pve_correction,        outputnode, [("out_files", "out_files")]),
         (pve_correction,        outputnode, [("results_numpy_npz", "pet_results_npz")]),
         (pve_correction,        outputnode, [("results_matlab_mat", "pet_results_mat")]),
         (applyxfm_CorrectedPET, outputnode, [("out_file", "corrected_pet_to_t1")]),
         (coregister,            outputnode, [("out_file", "pet_to_t1")]),
         ])

    return workflow
