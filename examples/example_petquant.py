import os
import os.path as op
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from coma.workflows.pet import create_pet_quantification_wf

from coma.datasets import sample
data_path = sample.data_path()

subjects_dir = op.join(data_path, "subjects")
output_dir = op.abspath('example_petquant')

info = dict(pet_image=[['subject_id', 'subject_id', 'pet']],
            roi_image=[['subject_id', '*DMN_rois']],
            t1_image=[['subject_id', '*T1brain']])

subject_list = ['Bend1']

infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                     name="infosource")
infosource.iterables = ('subject_id', subject_list)
datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=info.keys()),
                     name='datasource')

datasource.inputs.template = "%s/%s"
datasource.inputs.base_directory = data_path
datasource.inputs.field_template = dict(pet_image='data/%s/%s_%s.nii.gz',
                                        roi_image='data/%s/%s.nii.gz',
                                        t1_image='data/%s/%s.nii.gz')
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = True

datasink = pe.Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.overwrite = True

petquant = create_pet_quantification_wf("example_petquant")

workflow = pe.Workflow(name='ex_petquant')
workflow.base_dir = output_dir
workflow.connect([(infosource, datasource, [('subject_id', 'subject_id')])])

workflow.connect(
    [(infosource, petquant, [('subject_id', 'inputnode.subject_id')])])
workflow.connect([(datasource, petquant, [('pet_image', 'inputnode.pet')])])
workflow.connect([(datasource, petquant, [('roi_image', 'inputnode.rois')])])
workflow.connect([(datasource, petquant, [('t1_image', 'inputnode.t1')])])

workflow.connect(
    [(petquant, datasink, [("outputnode.out_files", "@subject_id.out_files"),
                           ("outputnode.corrected_pet_to_t1",
                            "@subject_id.corrected_pet_to_t1"),
                           ("outputnode.pet_results_npz",
                            "@subject_id.pet_results_npz"),
                           ])])

workflow.connect([(infosource, datasink, [('subject_id', '@subject_id')])])
workflow.write_graph()
workflow.config['execution'] = {'remove_unnecessary_outputs': 'false',
                                'hash_method': 'timestamp'}
workflow.run()
#workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 4})
