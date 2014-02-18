import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import os.path as op
import nipype.pipeline.engine as pe          # pypeline engine
from coma.workflows.denoised import create_denoised_timecourse_workflow

from coma.datasets import sample
data_path = sample.data_path()

subjects_dir = op.join(data_path,"subjects_normalized")
output_dir = op.abspath('denoised_timecourse')

info = dict(segmentation_file=[['subject_id', '*_scale500']],
            functional_images=[['subject_id', '*swvmsrf*']])

subject_list = ['Bend1']

infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                     name="infosource")
infosource.iterables = ('subject_id', subject_list)
datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=info.keys()),
                     name='datasource')

datasource.inputs.template = "%s/%s"
datasource.inputs.base_directory = data_dir
datasource.inputs.field_template = dict(fdg_pet_image='%s/.nii')
datasource.inputs.field_template = dict(functional_images='%s/restMotionProcessed/%s.img',
    segmentation_file="%s/%s.nii.gz")
datasource.inputs.template_args = info

denoised = create_denoised_timecourse_workflow("denoised")
denoised.inputnode.inputs.repetition_time = 2.4

datasink = pe.Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir

workflow = pe.Workflow(name='denoised')
workflow.base_dir = output_dir

workflow.connect([(infosource, datasource, [('subject_id', 'subject_id')])])
workflow.connect([(infosource, datasink, [('subject_id', '@subject_id')])])

workflow.connect([(infosource, denoised, [('subject_id', 'inputnode.subject_id')])])
workflow.connect([(datasource, denoised, [('segmentation_file', 'inputnode.segmentation_file')])])
workflow.connect([(datasource, denoised, [('functional_images', 'inputnode.functional_images')])])

workflow.connect(
    [(denoised, datasink, [('outputnode.stats_file', '@subject_id.stats_file')])])

workflow.run()
#workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 3})
