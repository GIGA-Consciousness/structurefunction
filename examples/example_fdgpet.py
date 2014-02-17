import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import os, os.path as op
import nipype.algorithms.misc as misc
import nipype.interfaces.cmtk as cmtk
import nipype.interfaces.freesurfer as fs
import nipype.pipeline.engine as pe          # pypeline engine
from nipype.workflows.rsfmri.coma.fmri_graphs import create_fmri_graphs, create_fmri_graph_grouping_workflow
from nipype.workflows.smri.cmtk.parcellate import create_parcellation_workflow
import cmp
cmp_config = cmp.configuration.PipelineConfiguration()
cmp_config.parcellation_scheme = "Lausanne2008"
parcellation_name = 'scale500'

data_dir = '/media/BlackBook/ERIKPETDTIFMRI/ControlsPETDTI'
subjects_dir = '/media/BlackBook/ERIKPETDTIFMRI/subjects_normalized/controls'
output_dir = '/media/BlackBook/ERIKPETDTIFMRI/fmrigraphs'

info = dict(functional_images=[['subject_id', '*swvmsrf*']],
            fmri_ICA_timecourse=[['subject_id','*timecourses*']],
            fmri_ICA_maps=[['subject_id','*component_ica*']],
            ica_mask_image=[['subject_id','*icaAnaMask*']],
            fdg_pet_image=[['subject_id','*fswp*']],)

subject_list = ['Bend1', 'Caho3', 'Cont4','Luyp5','Squi6','Tous7', 'Vanl8','Zeev9','Sauveur10','Tamblez11','Lambrech12','Foidart13','Geron14','Lambert15','Wathelet16']
#subject_list = ['Cont4','Luyp5']

infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                     name="infosource")
infosource.iterables = ('subject_id', subject_list)
datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=info.keys()),
                     name = 'datasource')

datasource.inputs.template = "%s/%s"
datasource.inputs.base_directory = data_dir
datasource.inputs.field_template = dict(functional_images='%s/restMotionProcessed/%s.img', fmri_ICA_timecourse='%s/restICA/components/%s.img', fmri_ICA_maps='%s/restICA/components/%s.img', ica_mask_image='%s/restICA/%s.img')
datasource.inputs.template_args = info

parcellate = create_parcellation_workflow("parcellate")
parcellate.inputs.inputnode.subjects_dir = subjects_dir

resample_fdg_pet = pe.Node(interface=fs.MRIConvert(), name='resample_fdg_pet')
resample_fdg_pet.inputs.out_type = 'nii'
regional_metabolism = pe.Node(interface=cmtk.RegionalValues(), name="regional_metabolism")
regional_metabolism.inputs.out_stats_file = 'regional_fdg_uptake.mat'

funky = create_fmri_graphs("funkytest")
funky.inputs.inputnode.resolution_network_file = cmp_config._get_lausanne_parcellation('Lausanne2008')[parcellation_name]['node_information_graphml']
funky.inputs.inputnode.repetition_time = 2.4

datasink = pe.Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.raise_on_empty = False
datasink.inputs.ignore_exception = True

graph = pe.Workflow(name='fmrigraphs')
graph.base_dir = output_dir
graph.connect([(infosource, datasource,[('subject_id', 'subject_id')])])
graph.connect([(infosource, funky,[('subject_id', 'inputnode.subject_id')])])
graph.connect([(infosource, parcellate,[('subject_id', 'inputnode.subject_id')])])
graph.connect([(parcellate, funky,[('outputnode.rois', 'inputnode.segmentation_file')])])
graph.connect([(datasource, funky,[('functional_images', 'inputnode.functional_images')])])

graph.connect([(parcellate, regional_metabolism,[('outputnode.rois', 'segmentation_file')])])
graph.connect([(parcellate, resample_fdg_pet,[('outputnode.rois', 'reslice_like')])])
graph.connect([(datasource, resample_fdg_pet,[('fdg_pet_image', 'in_file')])])
graph.connect([(resample_fdg_pet, regional_metabolism,[('out_file', 'in_files')])])
graph.connect([(funky, regional_metabolism,[('func_ntwk.CreateNodes.node_network', 'resolution_network_file')])])

graph.connect([(datasource, funky,[('fmri_ICA_timecourse', 'inputnode.fmri_ICA_timecourse')])])
graph.connect([(datasource, funky,[('fmri_ICA_maps', 'inputnode.fmri_ICA_maps')])])
graph.connect([(datasource, funky,[('ica_mask_image', 'inputnode.ica_mask_image')])])
graph.connect([(funky, datasink, [("outputnode.correlation_cff", "@cff.correlation"),
                                          ("outputnode.anticorrelation_cff", "@cff.anticorrelation"),
                                          ("outputnode.neuronal_cff", "@cff.neuronal"),
                                          ("outputnode.neuronal_ntwks", "@ntwks.neuronal"),
                                          ("outputnode.neuronal_regional_timecourse_stats", "@subject_id.denoised_fmri_timecourse"),
                                          ("outputnode.correlation_ntwks", "@ntwks.correlation"),
                                          ("outputnode.correlation_stats", "@subject_id.correlation_stats"),
                                          #("outputnode.anticorrelation_ntwks", "@ntwks.anticorrelation"),
                                          ("outputnode.anticorrelation_stats", "@subject_id.anticorrelation_stats"),
                                          ("outputnode.matching_stats", "@subject_id.matching_stats"),
                                          ])])
datasink.inputs.regexp_substitutions = [('_group_graphs_corr[\d]*/', '')]
graph.connect([(infosource, datasink,[('subject_id','@subject_id')])])
graph.connect([(regional_metabolism, datasink,[('networks','@subject_id.regional_metabolism')])])
graph.connect([(regional_metabolism, datasink,[('stats_file','@subject_id.regional_metabolic_stats')])])
#regional_metabolism.overwrite = True
import ipdb
#ipdb.set_trace()
#graph.run()
#graph.run(plugin='MultiProc', plugin_args={'n_procs' : 3})

averaging = create_fmri_graph_grouping_workflow(data_dir, output_dir, 'averaging')
averaging.base_dir = output_dir
averaging.inputs.l2inputnode.resolution_network_file = '/media/Mobsol/Parkinsons/subj1gpickle.pck'
#averaging.run()
averaging.run(plugin='MultiProc', plugin_args={'n_procs' : 3})
