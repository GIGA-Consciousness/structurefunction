import os
import os.path as op
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl
from nipype.workflows.dmri.fsl.epi import create_motion_correct_pipeline, create_eddy_correct_pipeline

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from coma.workflows.precoth import return_subject_data, create_precoth_pipeline_step1, create_precoth_pipeline_step2

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
  bvecs='data/%s/%s', bvals='data/%s/%s',
  fdg_pet_image='data/%s/%s.nii.gz')
datasource_step1.inputs.template_args = info
datasource_step1.inputs.sort_filelist = True

get_subject_data_interface = util.Function(input_names=["subject_id", "data_file"],
  output_names=["dose", "weight", "delay", "glycemie", "scan_time"], function=return_subject_data)
grab_subject_data = pe.Node(interface=get_subject_data_interface, name='grab_subject_data')
grab_subject_data.inputs.data_file = op.join(data_path, "SubjectData.csv")

datasink_step1 = pe.Node(interface=nio.DataSink(), name="datasink")
datasink_step1.inputs.base_directory = output_dir
datasink_step1.overwrite = True

step1 = create_precoth_pipeline_step1("precoth_step1", reg_pet_T1=True)
step1.inputs.inputnode.subjects_dir = subjects_dir


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
workflow.connect([(motioncorrect, step1,[('outputnode.out_bvec', 'inputnode.bvecs')])])

workflow.connect([(datasource_step1, step1,[('bvals', 'inputnode.bvals')])])
workflow.connect([(infosource, step1,[('subject_id', 'inputnode.subject_id')])])
workflow.connect([(datasource_step1, step1,[('fdg_pet_image', 'inputnode.fdgpet')])])

workflow.connect([(infosource, datasink_step1,[('subject_id','@subject_id')])])
workflow.connect([(step1, datasink_step1, [("outputnode.fa", "@subject_id.fa"),
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

workflow.connect([(eddycorrect, datasink_step1,[('outputnode.eddy_corrected', '@subject_id.corrected_dwi')])])
workflow.write_graph()
workflow.run()
#workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 4})




step2_dir  = op.abspath('precoth_twostep_step2')

step2 = create_precoth_pipeline_step2("precoth_step2")
lmax = 6
step2.inputs.csdeconv.maximum_harmonic_order = lmax
step2.inputs.estimateresponse.maximum_harmonic_order = lmax
step2.inputs.fsl2mrtrix.invert_x = True
step2.inputs.fsl2mrtrix.invert_y = False
step2.inputs.inputnode.subjects_dir = subjects_dir
step2.inputs.thalamus2precuneus2cortex.resolution_network_file = res_ntwk_file


info_step2 = dict(eddycorrected_dwi=[['subject_id']],
                fdg_pet_image=[['subject_id']],
                fa=[['subject_id']],
                single_fiber_mask=[['subject_id']],
                t1_brain=[['subject_id']],
                wm_mask=[['subject_id']],
                term_mask=[['subject_id']],
                rois=[['subject_id']])

datasource_step2 = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=info_step2.keys()),
                     name = 'datasource')

datasource_step2.inputs.template = "%s/%s"
datasource_step2.inputs.base_directory = step1_dir
datasource_step2.inputs.field_template = dict(eddycorrected_dwi='eddycorrected_dwi/*%s/*.nii.gz',
                                              fdg_pet_image='fdgpet/*%s/*.nii.gz',
                                              fa='fa/*%s/*.nii.gz',
                                              single_fiber_mask='single_fiber_mask/*%s/*.nii.gz',
                                              t1_brain='t1_brain/*%s/*.nii.gz',
                                              wm_mask='wm_mask/*%s/*.nii.gz',
                                              term_mask='term_mask/*%s/*.nii.gz',
                                              rois='rois/*%s/*.nii.gz',
                                              )
datasource_step2.inputs.template_args = info_step2
datasource_step2.inputs.sort_filelist = True

datasink_step2 = pe.Node(interface=nio.DataSink(), name="datasink")
datasink_step2.inputs.base_directory = output_dir
datasink_step2.overwrite = True

workflow2 = pe.Workflow(name='ex_precoth2')
workflow2.base_dir = output_dir
workflow2.connect([(infosource, datasource_step1,[('subject_id', 'subject_id')])])
workflow2.connect([(infosource, datasource_step2,[('subject_id', 'subject_id')])])

workflow.connect([(motioncorrect, step2,[('outputnode.out_bvec', 'inputnode.bvecs')])])
workflow2.connect([(datasource_step1, step2,[('bvals', 'inputnode.bvals')])])

workflow2.connect([(datasource_step2, step2,[('eddycorrected_dwi', 'inputnode.dwi')])])
workflow2.connect([(datasource_step2, step2,[('fdg_pet_image', 'inputnode.fdgpet')])])
workflow2.connect([(datasource_step2, step2,[('fa', 'inputnode.fa')])])
workflow2.connect([(datasource_step2, step2,[('single_fiber_mask', 'inputnode.single_fiber_mask')])])
workflow2.connect([(datasource_step2, step2,[('t1_brain', 'inputnode.t1_brain')])])
workflow2.connect([(datasource_step2, step2,[('wm_mask', 'inputnode.wm_mask')])])
workflow2.connect([(datasource_step2, step2,[('term_mask', 'inputnode.term_mask')])])
workflow2.connect([(datasource_step2, step2,[('rois', 'inputnode.rois')])])

workflow2.connect([(infosource, step2,[('subject_id', 'inputnode.subject_id')])])

workflow2.connect([(infosource, grab_subject_data,[('subject_id', 'subject_id')])])
workflow2.connect([(grab_subject_data, step2,[('dose', 'inputnode.dose')])])
workflow2.connect([(grab_subject_data, step2,[('weight', 'inputnode.weight')])])
workflow2.connect([(grab_subject_data, step2,[('delay', 'inputnode.delay')])])
workflow2.connect([(grab_subject_data, step2,[('glycemie', 'inputnode.glycemie')])])
workflow2.connect([(grab_subject_data, step2,[('scan_time', 'inputnode.scan_time')])])

workflow2.connect([(infosource, datasink_step2,[('subject_id','@subject_id')])])
workflow2.connect([(step2, datasink_step2, [("outputnode.csdeconv", "@subject_id.csdeconv"),
                                          ("outputnode.tracts_tck", "@subject_id.tracts_tck"),
                                          ("outputnode.summary", "@subject_id.summary"),
                                          ("outputnode.filtered_tractographies", "@subject_id.filtered_tractographies"),
                                          ("outputnode.matrix_file", "@subject_id.matrix_file"),
                                          ("outputnode.connectome", "@subject_id.connectome"),
                                          ("outputnode.CMR_nodes", "@subject_id.CMR_nodes"),
                                          ("outputnode.fa_t1space", "@subject_id.fa_t1space"),
                                          ])])
#workflow2.run()