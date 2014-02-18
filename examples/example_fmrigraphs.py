import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import os.path as op
import nipype.interfaces.cmtk as cmtk
import nipype.pipeline.engine as pe          # pypeline engine
from nipype.workflows.rsfmri.coma.fmri_graphs import create_fmri_graphs, create_fmri_graph_grouping_workflow

from coma.datasets import sample
data_path = sample.data_path()

subjects_dir = op.join(data_path,"subjects_normalized")
output_dir = op.abspath('fmrigraphs')

info = dict(functional_images=[['subject_id', '*swvmsrf*']],
            fmri_ICA_timecourse=[['subject_id','*timecourses*']],
            fmri_ICA_maps=[['subject_id','*component_ica*']],
            ica_mask_image=[['subject_id','*icaAnaMask*']])

subject_list = ['Bend1']

infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                     name="infosource")
infosource.iterables = ('subject_id', subject_list)
datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=info.keys()),
                     name = 'datasource')

datasource.inputs.template = "%s/%s"
datasource.inputs.base_directory = data_path
datasource.inputs.field_template = dict(functional_images='data/%s/restMotionProcessed/%s.img',
   fmri_ICA_timecourse='data/%s/restICA/components/%s.img', fmri_ICA_maps='data/%s/restICA/components/%s.img',
   ica_mask_image='data/%s/restICA/%s.img')
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = True

parcellate = pe.Node(interface=cmtk.Parcellate(), name="parcellate")
parcellate.inputs.subjects_dir = subjects_dir

funky = create_fmri_graphs("funkytest")

import cmp
cmp_config = cmp.configuration.PipelineConfiguration()
cmp_config.parcellation_scheme = "Lausanne2008"
parcellation_name = 'scale500'

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
graph.connect([(infosource, parcellate,[('subject_id', 'subject_id')])])

graph.connect([(datasource, funky,[('functional_images', 'inputnode.functional_images')])])
graph.connect([(datasource, funky,[('fmri_ICA_timecourse', 'inputnode.fmri_ICA_timecourse')])])
graph.connect([(datasource, funky,[('fmri_ICA_maps', 'inputnode.fmri_ICA_maps')])])
graph.connect([(datasource, funky,[('ica_mask_image', 'inputnode.ica_mask_image')])])
graph.connect([(parcellate, funky,[('roi_file', 'inputnode.segmentation_file')])])

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

graph.run()
#graph.run(plugin='MultiProc', plugin_args={'n_procs' : 3})

#averaging = create_fmri_graph_grouping_workflow(data_dir, output_dir, 'averaging')
#averaging.base_dir = output_dir
#averaging.inputs.l2inputnode.resolution_network_file = '/media/Mobsol/Parkinsons/subj1gpickle.pck'
#averaging.run()
#averaging.run(plugin='MultiProc', plugin_args={'n_procs' : 3})
