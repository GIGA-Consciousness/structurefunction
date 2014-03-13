import os
import os.path as op
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl
from nipype.workflows.dmri.fsl.epi import create_motion_correct_pipeline, create_eddy_correct_pipeline

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from coma.workflows.precoth import return_subject_data, create_precoth_pipeline_step1, create_precoth_pipeline_step3

from coma.datasets import sample
data_path = sample.data_path()

subjects_dir = op.join(data_path,"subjects")
step1_dir = op.abspath('precoth_twostep_step1')
output_dir = op.abspath('precoth_twostep_step1')

info = dict(dwi=[['subject_id', 'dwi']],
                bvecs=[['subject_id','bvecs']],
                bvals=[['subject_id','bvals']],
                fdg_pet_image=[['subject_id','petmr']])

subject_list = ['Bend1']

res_ntwk_file = op.join(os.environ['COMA_DIR'],'etc','precoth.graphml')




infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                     name="infosource")
infosource.iterables = ('subject_id', subject_list)
datasource_step1 = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=info.keys()),
                     name = 'datasource')

datasource_step1.inputs.template = "%s/%s"
datasource_step1.inputs.base_directory = data_path
datasource_step1.inputs.field_template = dict(dwi='data/%s/%s.nii.gz',
  bvecs='data/%s/%s', bvals='data/%s/%s')
datasource_step1.inputs.template_args = info
datasource_step1.inputs.sort_filelist = True

get_subject_data_interface = util.Function(input_names=["subject_id", "data_file"],
  output_names=["dose", "weight", "delay", "glycemie", "scan_time"], function=return_subject_data)
grab_subject_data = pe.Node(interface=get_subject_data_interface, name='grab_subject_data')
grab_subject_data.inputs.data_file = op.join(data_path, "SubjectData.csv")

datasink_step1 = pe.Node(interface=nio.DataSink(), name="datasink")
datasink_step1.inputs.base_directory = output_dir
datasink_step1.overwrite = True

workflow = pe.Workflow(name='ex_precoth1')
workflow.base_dir = output_dir
workflow.connect([(infosource, datasource_step1,[('subject_id', 'subject_id')])])

motioncorrect = create_motion_correct_pipeline(name='motioncorrect')
motioncorrect.inputs.inputnode.ref_num = 0
eddycorrect = create_eddy_correct_pipeline(name='eddycorrect')
eddycorrect.inputs.inputnode.ref_num = 0

workflow.connect([(datasource_step1, motioncorrect,[('dwi', 'inputnode.in_file')])])
workflow.connect([(motioncorrect, eddycorrect,[('outputnode.motion_corrected', 'inputnode.in_file')])])
workflow.connect([(eddycorrect, step1,[('outputnode.eddy_corrected', 'inputnode.dwi')])])
workflow.connect([(eddycorrect, datasink_step1,[('outputnode.eddy_corrected', '@subject_id.corrected_dwi')])])
workflow.write_graph()
#workflow.run()
#workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 4})




workflow2 = pe.Workflow(name='ex_precoth2')
workflow2.base_dir = output_dir

datasource_step2 = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=info.keys()),
                     name = 'datasource')

datasource_step2.inputs.template = "%s/%s"
datasource_step2.inputs.base_directory = data_path
datasource_step2.inputs.field_template = dict(dwi='data/%s/%s.nii.gz',
  bvecs='data/%s/%s', bvals='data/%s/%s',
  fdg_pet_image='data/%s/%s.nii.gz')
datasource_step2.inputs.template_args = info
datasource_step2.inputs.sort_filelist = True

datasink_step2 = pe.Node(interface=nio.DataSink(), name="datasink")
datasink_step2.inputs.base_directory = output_dir
datasink_step2.overwrite = True


step2_dir  = op.abspath('precoth_twostep_step2')
step2 = create_precoth_pipeline_step1("precoth_step2", reg_pet_T1=True)
step2.inputs.inputnode.subjects_dir = subjects_dir

workflow2.connect([(infosource, datasource_step2,[('subject_id', 'subject_id')])])
workflow2.connect([(datasource_step2, step2,[('bvals', 'inputnode.bvals')])])
workflow2.connect([(datasource_step2, step2,[('bvecs', 'inputnode.bvecs')])])
workflow2.connect([(infosource, step2,[('subject_id', 'inputnode.subject_id')])])
workflow2.connect([(datasource_step2, step2,[('fdg_pet_image', 'inputnode.fdgpet')])])
workflow2.connect([(infosource, datasink_step2,[('subject_id','@subject_id')])])
workflow2.connect([(step2, datasink_step2, [("outputnode.fa", "@subject_id.fa"),
                                          ("outputnode.md", "@subject_id.md"),
                                          ("outputnode.rois", "@subject_id.rois"),
                                          ("outputnode.rgb_fa", "@subject_id.rgb_fa"),
                                          ("outputnode.single_fiber_mask", "@subject_id.single_fiber_mask"),
                                          ("outputnode.t1", "@subject_id.t1"),
                                          ("outputnode.t1_brain", "@subject_id.t1_brain"),
                                          ("outputnode.fdgpet", "@subject_id.fdgpet"),
                                          ("outputnode.wm_mask", "@subject_id.wm_mask"),
                                          ("outputnode.term_mask", "@subject_id.term_mask"),
                                          ])])





step3_dir  = op.abspath('precoth_twostep_step3')

step3 = create_precoth_pipeline_step2("precoth_step3")
lmax = 4
step3.inputs.csdeconv.maximum_harmonic_order = lmax
step3.inputs.estimateresponse.maximum_harmonic_order = lmax
step3.inputs.fsl2mrtrix.invert_x = True
step3.inputs.fsl2mrtrix.invert_y = False
step3.inputs.inputnode.subjects_dir = subjects_dir
step3.inputs.thalamus2precuneus2cortex.resolution_network_file = res_ntwk_file


info_step3 = dict(corrected_dwi=[['subject_id']],
                corrected_bvals=[['subject_id']],
                corrected_bvecs=[['subject_id']],
                corrected_fdg_pet_image=[['subject_id']],
                corrected_fa=[['subject_id']],
                corrected_single_fiber_mask=[['subject_id']],
                corrected_t1_brain=[['subject_id']],
                corrected_wm_mask=[['subject_id']],
                corrected_term_mask=[['subject_id']],
                corrected_rois=[['subject_id']])

datasource_step3 = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=info_step3.keys()),
                     name = 'datasource')

datasource_step3.inputs.template = "%s/%s"
datasource_step3.inputs.base_directory = step1_dir
datasource_step3.inputs.field_template = dict(corrected_dwi='corrected_dwi/*%s/*.nii.gz',
                                              corrected_bvals='corrected_bvals/*%s/*.bval', 
                                              corrected_bvecs='corrected_bvecs/*%s/*.bvec',
                                              corrected_fdg_pet_image='fdgpet/*%s/*.nii.gz',
                                              corrected_fa='fa/*%s/*.nii.gz',
                                              corrected_single_fiber_mask='corrected_single_fiber_mask/*%s/*.nii.gz',
                                              corrected_t1_brain='corrected_t1_brain/*%s/*.nii.gz',
                                              corrected_wm_mask='corrected_wm_mask/*%s/*.nii.gz',
                                              corrected_term_mask='corrected_term_mask/*%s/*.nii.gz',
                                              corrected_rois='corrected_rois/*%s/*.nii.gz',
                                              )
datasource_step3.inputs.template_args = info_step3
datasource_step3.inputs.sort_filelist = True

datasink_step3 = pe.Node(interface=nio.DataSink(), name="datasink")
datasink_step3.inputs.base_directory = output_dir
datasink_step3.overwrite = True

workflow3 = pe.Workflow(name='ex_precoth2')
workflow3.base_dir = output_dir
workflow3.connect([(infosource, datasource_step3,[('subject_id', 'subject_id')])])

workflow3.connect([(datasource_step3, step3,[('corrected_bvals', 'inputnode.bvals')])])
workflow3.connect([(datasource_step3, step3,[('corrected_bvecs', 'inputnode.bvecs')])])

workflow3.connect([(datasource_step3, step3,[('corrected_dwi', 'inputnode.dwi')])])
workflow3.connect([(datasource_step3, step3,[('corrected_fdg_pet_image', 'inputnode.fdgpet')])])
workflow3.connect([(datasource_step3, step3,[('corrected_fa', 'inputnode.fa')])])
workflow3.connect([(datasource_step3, step3,[('corrected_single_fiber_mask', 'inputnode.single_fiber_mask')])])
workflow3.connect([(datasource_step3, step3,[('corrected_t1_brain', 'inputnode.t1_brain')])])
workflow3.connect([(datasource_step3, step3,[('corrected_wm_mask', 'inputnode.wm_mask')])])
workflow3.connect([(datasource_step3, step3,[('corrected_term_mask', 'inputnode.term_mask')])])
workflow3.connect([(datasource_step3, step3,[('corrected_rois', 'inputnode.rois')])])

workflow3.connect([(infosource, step3,[('subject_id', 'inputnode.subject_id')])])

workflow3.connect([(infosource, grab_subject_data,[('subject_id', 'subject_id')])])
workflow3.connect([(grab_subject_data, step3,[('dose', 'inputnode.dose')])])
workflow3.connect([(grab_subject_data, step3,[('weight', 'inputnode.weight')])])
workflow3.connect([(grab_subject_data, step3,[('delay', 'inputnode.delay')])])
workflow3.connect([(grab_subject_data, step3,[('glycemie', 'inputnode.glycemie')])])
workflow3.connect([(grab_subject_data, step3,[('scan_time', 'inputnode.scan_time')])])

workflow3.connect([(infosource, datasink_step3,[('subject_id','@subject_id')])])
workflow3.connect([(step3, datasink_step3, [("outputnode.csdeconv", "@subject_id.csdeconv"),
                                          ("outputnode.tracts_tck", "@subject_id.tracts_tck"),
                                          ("outputnode.summary", "@subject_id.summary"),
                                          ("outputnode.filtered_tractographies", "@subject_id.filtered_tractographies"),
                                          ("outputnode.matrix_file", "@subject_id.matrix_file"),
                                          ("outputnode.connectome", "@subject_id.connectome"),
                                          ("outputnode.CMR_nodes", "@subject_id.CMR_nodes"),
                                          ("outputnode.fa_t1space", "@subject_id.fa_t1space"),
                                          ])])
#workflow3.run()