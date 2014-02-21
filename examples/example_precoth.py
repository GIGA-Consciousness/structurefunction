import os
import os.path as op
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine

import nipype.interfaces.fsl as fsl
from nipype.workflows.dmri.fsl.dti import create_eddy_correct_pipeline

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from coma.workflows.precoth import create_precoth_pipeline

from coma.datasets import sample
data_path = sample.data_path()

subjects_dir = op.join(data_path,"subjects")
output_dir = op.abspath('precoth')

info = dict(dwi=[['subject_id', 'dwi']],
                bvecs=[['subject_id','bvecs']],
                bvals=[['subject_id','bvals']],
                fdg_pet_image=[['subject_id','petmr']])

subject_list = ['Bend1']

res_ntwk_file = op.join(os.environ['COMA_DIR'],'etc','precoth.graphml')

infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                     name="infosource")
infosource.iterables = ('subject_id', subject_list)
datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=info.keys()),
                     name = 'datasource')

datasource.inputs.template = "%s/%s"
datasource.inputs.base_directory = data_path
datasource.inputs.field_template = dict(dwi='data/%s/%s.nii.gz',
  bvecs='data/%s/%s', bvals='data/%s/%s',
  fdg_pet_image='data/%s/%s.nii')
datasource.inputs.template_args = info

datasink = pe.Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir


dti = create_precoth_pipeline("coma_precoth")

lmax = 6
dti.inputs.csdeconv.maximum_harmonic_order = lmax
dti.inputs.estimateresponse.maximum_harmonic_order = lmax
dti.inputs.fsl2mrtrix.invert_x = True
dti.inputs.fsl2mrtrix.invert_y = False
dti.inputs.inputnode.subjects_dir = subjects_dir
dti.inputs.thalamus2precuneus2cortex.resolution_network_file = res_ntwk_file
dti.inputs.fdgpet_regions.resolution_network_file = res_ntwk_file

workflow = pe.Workflow(name='ex_precoth')
workflow.base_dir = output_dir
workflow.connect([(infosource, datasource,[('subject_id', 'subject_id')])])

eddycorrect = create_eddy_correct_pipeline(name='eddycorrect')
eddycorrect.inputs.inputnode.ref_num = 1

workflow.connect([(datasource, eddycorrect,[('dwi', 'inputnode.in_file')])])
workflow.connect([(eddycorrect, dti,[('outputnode.eddy_corrected', 'inputnode.dwi')])])
workflow.connect([(infosource, dti,[('subject_id', 'inputnode.subject_id')])])
workflow.connect([(datasource, dti,[('fdg_pet_image', 'inputnode.fdgpet')])])
workflow.connect([(datasource, dti,[('bvecs', 'inputnode.bvecs')])])
workflow.connect([(datasource, dti,[('bvals', 'inputnode.bvals')])])

workflow.connect([(dti, datasink, [("outputnode.fa", "@subject_id.fa"),
                                          ("outputnode.md", "@subject_id.md"),
                                          ])])

workflow.connect([(infosource, datasink,[('subject_id','@subject_id')])])
workflow.write_graph()
workflow.run()
