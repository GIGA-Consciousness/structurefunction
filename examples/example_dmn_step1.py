import os.path as op
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from coma.workflows.dmnwf import create_dmn_pipeline_step1

from coma.datasets import sample
data_path = sample.data_path()

name = 'example_dmn_step1'

subjects_dir = op.join(data_path,"subjects")

info = dict(dwi=[['subject_id', 'dwi']],
            bvecs=[['subject_id', 'bvecs']],
            bvals=[['subject_id', 'bvals']],
            fdgpet=[['subject_id', 'petmr']])

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
datasource.inputs.field_template = dict(dwi='data/%s/%s.nii.gz',
  bvecs='data/%s/%s', bvals='data/%s/%s', fdgpet='data/%s/%s.nii.gz')
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = True

datasink = pe.Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = op.abspath(name)

dmnwf = create_dmn_pipeline_step1()

workflow = pe.Workflow(name=name)
workflow.base_dir = name

workflow.connect([(infosource, datasource, [('subject_id', 'subject_id')])])
workflow.connect([(infosource, datasink, [('subject_id', '@subject_id')])])

workflow.connect([(infosource, dmnwf, [('subject_id', 'inputnode.subject_id')])])
workflow.connect([(infosource, dmnwf, [('subjects_dir', 'inputnode.subjects_dir')])])

workflow.connect([(datasource, dmnwf, [('dwi', 'inputnode.dwi')])])
workflow.connect([(datasource, dmnwf, [('bvecs', 'inputnode.bvecs')])])
workflow.connect([(datasource, dmnwf, [('bvals', 'inputnode.bvals')])])
workflow.connect([(datasource, dmnwf, [('fdgpet', 'inputnode.fdgpet')])])

workflow.connect([(dmnwf, datasink, [("outputnode.t1", "subject_id.@t1"),
                                       ("outputnode.wm_prob", "subject_id.@wm_prob"),
                                       ("outputnode.gm_prob", "subject_id.@gm_prob"),
                                       ("outputnode.csf_prob", "subject_id.@csf_prob"),
                                       ("outputnode.single_fiber_mask", "subject_id.@single_fiber_mask"),
                                       ("outputnode.fa", "subject_id.@fa"),
                                       ("outputnode.rgb_fa", "subject_id.@rgb_fa"),
                                       ("outputnode.md", "subject_id.@md"),
                                       ("outputnode.mode", "subject_id.@mode"),
                                       ("outputnode.t1_brain", "subject_id.@t1_brain"),
                                       ("outputnode.wm_mask", "subject_id.@wm_mask"),
                                       ("outputnode.term_mask", "subject_id.@term_mask"),
                                       ("outputnode.aparc_aseg", "subject_id.@aparc_aseg"),
                                       ("outputnode.tissue_class_files", "subject_id.@tissue_class_files"),
                                       ])])

workflow.connect([(dmnwf, datasink, [("outputnode.rois", "subject_id.@rois"),
                                       ("outputnode.rois_to_dwi", "subject_id.@rois_to_dwi"),
                                       ("outputnode.wmmask_to_dwi", "subject_id.@wmmask_to_dwi"),
                                       ("outputnode.termmask_to_dwi", "subject_id.@termmask_to_dwi"),
                                       ("outputnode.dwi_to_t1_matrix", "subject_id.@dwi_to_t1_matrix"),
                                       ])])

workflow.connect([(dmnwf, datasink, [("outputnode.pet_to_t1", "subject_id.@pet_to_t1"),
                                       ("outputnode.corrected_pet_to_t1", "subject_id.@corrected_pet_to_t1"),
                                       ("outputnode.pet_results_npz", "subject_id.@pet_results_npz"),
                                       ("outputnode.pet_results_mat", "subject_id.@pet_results_mat"),
                                       ])])

workflow.write_graph()
workflow.run()
#workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 3})
