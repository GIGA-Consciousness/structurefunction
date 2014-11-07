import os.path as op
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import cmp

import nipype.interfaces.fsl as fsl

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from coma.workflows.fsconnectivity import create_fsconnectivity_pipeline

from coma.datasets import sample
data_path = sample.data_path()

name = 'example_fsconnectivity'

subjects_dir = op.join(data_path,"subjects")

parcellation_name = 'scale33'
cmp_config = cmp.configuration.PipelineConfiguration()
cmp_config.parcellation_scheme = "Lausanne2008"


info = dict(dwi=[['subject_id', 'dwi']],
            bvecs=[['subject_id', 'bvecs']],
            bvals=[['subject_id', 'bvals']])

subject_list = ['Bend1']

infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id', 'subjects_dir']),
                     name="infosource")

infosource.iterables = ('subject_id', subject_list)
infosource.inputs.subjects_dir = subjects_dir
datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=info.keys()),
                     name='datasource')

datasource.inputs.template = "%s/%s"
datasource.inputs.base_directory = data_path
datasource.inputs.field_template = dict(dwi='data/%s/%s.nii.gz', bvecs='data/%s/%s', bvals='data/%s/%s')
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = True

datasink = pe.Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = op.abspath(name)

dwi = create_fsconnectivity_pipeline(parcellation_name=parcellation_name)

# For C.H.U. Liege Siemens Allegra 3T
dwi.inputs.tracking.fsl2mrtrix.invert_x = True
dwi.inputs.tracking.fsl2mrtrix.invert_y = False
dwi.inputs.tracking.fsl2mrtrix.invert_z = False

# Set a number of tracks
dwi.inputs.tracking.CSDstreamtrack.desired_number_of_tracks = 30000
dwi.inputs.tracking.CSDstreamtrack.minimum_tract_length = 10


workflow = pe.Workflow(name=name)
workflow.base_dir = name

workflow.connect([(infosource, datasource, [('subject_id', 'subject_id')])])
workflow.connect([(infosource, datasink, [('subject_id', '@subject_id')])])

workflow.connect([(infosource, dwi, [('subject_id', 'inputnode.subject_id')])])
workflow.connect([(infosource, dwi, [('subjects_dir', 'inputnode.subjects_dir')])])



dwi.inputs.inputnode.resolution_network_file = cmp_config._get_lausanne_parcellation('Lausanne2008')[parcellation_name]['node_information_graphml']

workflow.connect([(datasource, dwi, [('dwi', 'inputnode.dwi')])])
workflow.connect([(datasource, dwi, [('bvecs', 'inputnode.bvecs')])])
workflow.connect([(datasource, dwi, [('bvals', 'inputnode.bvals')])])

workflow.connect([(dwi, datasink, [("outputnode.t1_brain", "subject_id.@t1_brain"),
                                   ("outputnode.t1", "subject_id.@t1"),
                                   ("outputnode.wm_prob", "subject_id.@wm"),
                                   ("outputnode.gm_prob", "subject_id.@gm"),
                                   ("outputnode.csf_prob", "subject_id.@csf"),
                                   ("outputnode.rgb_fa", "subject_id.@rgb_fa"),
                                   ("outputnode.fa", "subject_id.@fa"),
                                   ("outputnode.md", "subject_id.@md"),
                                   ("outputnode.single_fiber_mask", "subject_id.@single_fiber_mask"),
                                   ("outputnode.wm_mask", "subject_id.@wm_mask"),
                                   ("outputnode.term_mask", "subject_id.@term_mask"),
                                   ("outputnode.aparc_aseg", "subject_id.@aparc_aseg"),
                                   ])])

workflow.config['execution'] = {'remove_unnecessary_outputs': 'false',
                                   'hash_method': 'timestamp'}
workflow.write_graph()
#workflow.run()
workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 11})
