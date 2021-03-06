import os.path as op
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from coma.workflows.dmnwf import create_dmn_pipeline_step1
from coma.helpers import return_subject_data

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

get_subject_data_interface = util.Function(input_names=["subject_id", "data_file"],
  output_names=["dose", "weight", "delay", "glycemie", "scan_time"], function=return_subject_data)
grab_subject_data = pe.Node(interface=get_subject_data_interface, name='grab_subject_data')
grab_subject_data.inputs.data_file = op.join(data_path, "SubjectData.csv")

datasink = pe.Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = op.abspath(name)
#datasink.inputs.parameterization = False
datasink.inputs.substitutions = [('_subject_id_', '')]


dmnwf = create_dmn_pipeline_step1()

workflow = pe.Workflow(name=name)
workflow.base_dir = name

workflow.connect([(infosource, datasource, [('subject_id', 'subject_id')])])
workflow.connect([(infosource, datasink, [('subject_id', '@subject_id')])])
workflow.connect([(infosource, grab_subject_data,[('subject_id', 'subject_id')])])

workflow.connect([(grab_subject_data, dmnwf,[('dose', 'inputnode.dose')])])
workflow.connect([(grab_subject_data, dmnwf,[('weight', 'inputnode.weight')])])
workflow.connect([(grab_subject_data, dmnwf,[('delay', 'inputnode.delay')])])
workflow.connect([(grab_subject_data, dmnwf,[('glycemie', 'inputnode.glycemie')])])
workflow.connect([(grab_subject_data, dmnwf,[('scan_time', 'inputnode.scan_time')])])

workflow.connect([(infosource, dmnwf, [('subject_id', 'inputnode.subject_id')])])
workflow.connect([(infosource, dmnwf, [('subjects_dir', 'inputnode.subjects_dir')])])

workflow.connect([(datasource, dmnwf, [('dwi', 'inputnode.dwi')])])
workflow.connect([(datasource, dmnwf, [('bvecs', 'inputnode.bvecs')])])
workflow.connect([(datasource, dmnwf, [('bvals', 'inputnode.bvals')])])
workflow.connect([(datasource, dmnwf, [('fdgpet', 'inputnode.fdgpet')])])

workflow.connect([(dmnwf, datasink, [("outputnode.t1", "Step1Results.@t1"),
                                       ("outputnode.wm_prob", "Step1Results.@wm_prob"),
                                       ("outputnode.gm_prob", "Step1Results.@gm_prob"),
                                       ("outputnode.csf_prob", "Step1Results.@csf_prob"),
                                       ("outputnode.single_fiber_mask", "Step1Results.@single_fiber_mask"),
                                       ("outputnode.fa", "Step1Results.@fa"),
                                       ("outputnode.rgb_fa", "Step1Results.@rgb_fa"),
                                       ("outputnode.md", "Step1Results.@md"),
                                       ("outputnode.mode", "Step1Results.@mode"),
                                       ("outputnode.t1_brain", "Step1Results.@t1_brain"),
                                       ("outputnode.wm_mask", "Step1Results.@wm_mask"),
                                       ("outputnode.term_mask", "Step1Results.@term_mask"),
                                       ("outputnode.aparc_aseg", "Step1Results.@aparc_aseg"),
                                       ("outputnode.tissue_class_files", "Step1Results.@tissue_class_files"),
                                       ("outputnode.t1_to_dwi", "Step1Results.@t1_to_dwi"),
                                       ])])

workflow.connect([(dmnwf, datasink, [("outputnode.rois", "Step1Results.@rois"),
                                       ("outputnode.rois_to_dwi", "Step1Results.@rois_to_dwi"),
                                       ("outputnode.wmmask_to_dwi", "Step1Results.@wmmask_to_dwi"),
                                       ("outputnode.termmask_to_dwi", "Step1Results.@termmask_to_dwi"),
                                       ("outputnode.dwi_to_t1_matrix", "Step1Results.@dwi_to_t1_matrix"),
                                       ("outputnode.single_fiber_mask_cortex_only", "Step1Results.@single_fiber_mask_cortex_only"),
                                       ])])

workflow.connect([(dmnwf, datasink, [("outputnode.SUV_corrected_pet_to_t1", "Step1Results.@SUV_corrected_pet_to_t1"),
                                       ("outputnode.AIF_corrected_pet_to_t1", "Step1Results.@AIF_corrected_pet_to_t1"),
                                       ("outputnode.pet_results_npz", "Step1Results.@pet_results_npz"),
                                       ("outputnode.pet_results_mat", "Step1Results.@pet_results_mat"),
                                       ("outputnode.orig_pet_to_t1", "Step1Results.@orig_pet_to_t1"),
                                       ])])


workflow.write_graph()
workflow.config['execution'] = {'remove_unnecessary_outputs': 'false',
                                   'hash_method': 'timestamp'}
workflow.run()
#workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 3})
