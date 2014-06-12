import os.path as op
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from coma.workflows.dti.basic import damaged_brain_dti_processing

from coma.datasets import sample
data_path = sample.data_path()

name = 'example_dwi_preproc'

subjects_dir = op.join(data_path,"subjects")

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

dwi = damaged_brain_dti_processing()

workflow = pe.Workflow(name=name)
workflow.base_dir = name

workflow.connect([(infosource, datasource, [('subject_id', 'subject_id')])])
workflow.connect([(infosource, datasink, [('subject_id', '@subject_id')])])

workflow.connect([(infosource, dwi, [('subject_id', 'inputnode.subject_id')])])
workflow.connect([(infosource, dwi, [('subjects_dir', 'inputnode.subjects_dir')])])

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
workflow.run()
#workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 3})
