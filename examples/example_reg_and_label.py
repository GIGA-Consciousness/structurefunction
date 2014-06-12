import os.path as op
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from coma.workflows.dmnwf import create_reg_and_label_wf

from coma.datasets import sample
data_path = sample.data_path()

name = 'example_reg_label'

subjects_dir = op.join(data_path,"subjects")

info = dict(aparc_aseg=[['subject_id', 'subject_id', 'aparc+aseg']],
            fa=[['subject_id', 'subject_id', 'fa']],
            wm_mask=[['subject_id', 'subject_id', 'wm_seed_mask']],
            term_mask=[['subject_id', 'subject_id', 'term_mask']])

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
datasource.inputs.field_template = dict(aparc_aseg='data/%s/%s_%s.nii.gz',
  fa='data/%s/%s_%s.nii.gz', wm_mask='data/%s/%s_%s.nii.gz', term_mask='data/%s/%s_%s.nii.gz')
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = True

datasink = pe.Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = op.abspath(name)
datasink.overwrite = True

regwf = create_reg_and_label_wf()

workflow = pe.Workflow(name=name)
workflow.base_dir = name

workflow.connect([(infosource, datasource, [('subject_id', 'subject_id')])])
workflow.connect([(infosource, datasink, [('subject_id', '@subject_id')])])

workflow.connect([(infosource, regwf, [('subject_id', 'inputnode.subject_id')])])
workflow.connect([(datasource, regwf, [('aparc_aseg', 'inputnode.aparc_aseg')])])
workflow.connect([(datasource, regwf, [('fa', 'inputnode.fa')])])
workflow.connect([(datasource, regwf, [('wm_mask', 'inputnode.wm_mask')])])
workflow.connect([(datasource, regwf, [('term_mask', 'inputnode.termination_mask')])])

workflow.connect([(regwf, datasink, [("outputnode.dwi_to_t1_matrix", "subject_id.@dwi_to_t1_matrix"),
                                   ("outputnode.t1_to_dwi_matrix", "subject_id.@t1_to_dwi_matrix"),
                                   ("outputnode.rois_to_dwi", "subject_id.@rois_to_dwi"),
                                   ("outputnode.rois", "subject_id.@rois"),
                                   ("outputnode.wmmask_to_dwi", "subject_id.@wmmask_to_dwi"),
                                   ("outputnode.termmask_to_dwi", "subject_id.@termmask_to_dwi"),
                                   ("outputnode.highres_t1_to_dwi_matrix", "subject_id.@highres_t1_to_dwi_matrix"),
                                   ])])
workflow.config['execution'] = {'remove_unnecessary_outputs': 'false',
                                   'hash_method': 'timestamp'}
workflow.run()
#workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 3})
