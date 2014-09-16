import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import os.path as op
import nipype.pipeline.engine as pe          # pypeline engine
from coma.workflows.connectivity import create_connectivity_pipeline

from coma.datasets import sample
data_path = sample.data_path()

import cmp
parcellation_name = 'scale33'
cmp_config = cmp.configuration.PipelineConfiguration()
cmp_config.parcellation_scheme = "Lausanne2008"

subjects_dir = op.join(data_path,"subjects")
output_dir = op.abspath('dwi_connectome')

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

structural = create_connectivity_pipeline("structural", parcellation_name)
structural.inputs.mapping.inputnode_within.resolution_network_file = cmp_config._get_lausanne_parcellation('Lausanne2008')[parcellation_name]['node_information_graphml']

# It's recommended to use low max harmonic order in damaged brains
lmax = 6
structural.inputs.mapping.csdeconv.maximum_harmonic_order = lmax
structural.inputs.mapping.estimateresponse.maximum_harmonic_order = lmax

# Required for scans from C.H.U Liege
structural.inputs.mapping.fsl2mrtrix.invert_x = True

datasink = pe.Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir

workflow = pe.Workflow(name='connnectivity')
workflow.base_dir = output_dir

workflow.connect([(infosource, datasource, [('subject_id', 'subject_id')])])
workflow.connect([(infosource, datasink, [('subject_id', '@subject_id')])])

workflow.connect([(infosource, structural, [('subject_id', 'inputnode.subject_id')])])
workflow.connect([(infosource, structural, [('subjects_dir', 'inputnode.subjects_dir')])])
workflow.connect([(datasource, structural, [('dwi', 'inputnode.dwi')])])
workflow.connect([(datasource, structural, [('bvecs', 'inputnode.bvecs')])])
workflow.connect([(datasource, structural, [('bvals', 'inputnode.bvals')])])

workflow.connect(
    [(structural, datasink, [('outputnode.cmatrix', '@subject_id.cmatrix')])])
workflow.connect(
    [(structural, datasink, [('outputnode.connectome', '@subject_id.connectome')])])

#workflow.run()
workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 6})
