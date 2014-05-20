import os
import os.path as op
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs
from nipype.workflows.misc.utils import select_aparc

fsl.FSLCommand.set_default_output_type('NIFTI')

from coma.interfaces.pve import PartialVolumeCorrection

def create_pet_quantification_wf(name="petquant"):
    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=["t1",
                                                 "wm",
                                                 "gm",
                                                 "csf",
                                                 "rois",
                                                 "pet"]),
        name="inputnode")


    coregister = pe.Node(interface=fsl.FLIRT(dof=6), name = 'coregister')
    coregister.inputs.cost = ('corratio')
    coregister.inputs.interp = 'trilinear'

    convertxfm = pe.Node(interface=fsl.ConvertXFM(), name = 'convertxfm')
    convertxfm.inputs.invert_xfm = True

    applyxfm_t1 = pe.Node(interface=fsl.ApplyXfm(), name = 'applyxfm_t1')
    applyxfm_t1.inputs.apply_xfm = True
    applyxfm_t1.inputs.interp = 'trilinear'

    applyxfm_gm = applyxfm_t1.clone("applyxfm_gm")
    applyxfm_gm.inputs.interp = 'nearestneighbour'
    applyxfm_wm = applyxfm_gm.clone("applyxfm_wm")
    applyxfm_csf = applyxfm_gm.clone("applyxfm_csf")

    applyxfm_rois = applyxfm_t1.clone("applyxfm_rois")
    applyxfm_rois.inputs.interp = 'nearestneighbour'

    pve_correction = pe.Node(interface=PartialVolumeCorrection(), name = 'pve_correction')
    pve_correction.inputs.skip_atlas = False
    pve_correction.inputs.use_fs_LUT = False

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    workflow.connect(
        [(inputnode, coregister, [('t1', 'reference')])])
    workflow.connect(
        [(inputnode, coregister, [('pet', 'in_file')])])
    workflow.connect(
        [(coregister, convertxfm, [('out_matrix_file', 'in_file')])])
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
        [(inputnode, applyxfm_gm, [('gm', 'in_file')])])

    workflow.connect(
        [(inputnode, applyxfm_wm, [('pet', 'reference')])])    
    workflow.connect(
        [(inputnode, applyxfm_wm, [('wm', 'in_file')])])
    
    workflow.connect(
        [(inputnode, applyxfm_csf, [('pet', 'reference')])])    
    workflow.connect(
        [(inputnode, applyxfm_csf, [('csf', 'in_file')])])

    workflow.connect(
        [(inputnode, applyxfm_rois, [('pet', 'reference')])])
    workflow.connect(
        [(inputnode, applyxfm_rois, [('rois', 'in_file')])])

    workflow.connect(
        [(applyxfm_t1, pve_correction, [('out_file', 't1_file')])])
    workflow.connect(
        [(inputnode, pve_correction, [('pet', 'pet_file')])])
    workflow.connect(
        [(applyxfm_gm, pve_correction, [('out_file', 'grey_matter_file')])])
    workflow.connect(
        [(applyxfm_wm, pve_correction, [('out_file', 'white_matter_file')])])
    workflow.connect(
        [(applyxfm_csf, pve_correction, [('out_file', 'csf_file')])])
    workflow.connect(
        [(applyxfm_rois, pve_correction, [('out_file', 'roi_file')])])

    output_fields = ["mueller_gartner_rousset", "pet_to_t1"]

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=output_fields),
        name="outputnode")

    workflow.connect(
        [(pve_correction, outputnode, [("mueller_gartner_rousset", "mueller_gartner_rousset")]),
         (coregister,     outputnode, [("out_file", "pet_to_t1")]),
         ])

    return workflow
