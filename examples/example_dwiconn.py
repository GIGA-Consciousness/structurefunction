import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import os.path as op
import nipype.pipeline.engine as pe          # pypeline engine
from coma.workflows.dti import create_connectivity_pipeline

from coma.datasets import sample
data_path = sample.data_path()

subjects_dir = op.join(data_path,"subjects")
output_dir = op.abspath('dwi_connectome')

info = dict(dwi=[['subject_id', '*DTI']],
            bvecs=[['subject_id', '*bvecs']],
            bvals=[['subject_id', '*bvals']])

subject_list = ['Bend1']

infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                     name="infosource")

infosource.iterables = ('subject_id', subject_list)
datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=info.keys()),
                     name='datasource')

datasource.inputs.template = "%s/%s"
datasource.inputs.base_directory = data_dir
datasource.inputs.field_template = dict(dwi='%s/%s.nii.gz', bvecs='%s/%s', bvals='%s/%s')
datasource.inputs.template_args = info

structural = create_connectivity_pipeline("structural")
#structural.inputnode.inputs.resolution_network_file = 
structural.inputs.inputnode.subjects_dir = subjects_dir

# It's recommended to use low max harmonic order in damaged brains
lmax = 6
structural.inputs.connectivity.mapping.csdeconv.maximum_harmonic_order = lmax
structural.inputs.connectivity.mapping.estimateresponse.maximum_harmonic_order = lmax

# Required for scans from C.H.U Liege
structural.inputs.connectivity.mapping.fsl2mrtrix.invert_x = True

datasink = pe.Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir

workflow = pe.Workflow(name='denoised')
workflow.base_dir = output_dir

workflow.connect([(infosource, datasource, [('subject_id', 'subject_id')])])
workflow.connect([(infosource, datasink, [('subject_id', '@subject_id')])])

workflow.connect([(infosource, structural, [('subject_id', 'inputnode.subject_id')])])
workflow.connect([(datasource, structural, [('dwi', 'inputnode.dwi')])])
workflow.connect([(datasource, structural, [('bvecs', 'inputnode.bvecs')])])
workflow.connect([(datasource, structural, [('bvals', 'inputnode.bvals')])])

workflow.connect(
    [(structural, datasink, [('outputnode.cmatrix', '@subject_id.cmatrix')])])
workflow.connect(
    [(structural, datasink, [('outputnode.connectome', '@subject_id.connectome')])])

workflow.run()
#workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 3})
