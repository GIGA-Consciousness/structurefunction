import os.path as op
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from coma.workflows.dmnwf import create_dmn_pipeline_step2

from coma.datasets import sample
data_path = sample.data_path()

name = 'example_dmn_step2'

subjects_dir = op.join(data_path,"subjects")

info = dict(dwi=[['subject_id', 'dwi']],
            bvecs=[['subject_id', 'bvecs']],
            bvals=[['subject_id', 'bvals']],
            single_fiber_mask=[['subject_id', 'subject_id', 'SingleFiberMask_dwi']],
            wm_mask=[['subject_id', 'subject_id', 'wm_seed_mask_dwi']],
            termination_mask=[['subject_id', 'subject_id', 'term_mask_dwi']],
            registration_matrix_file=[['subject_id', 'subject_id', 'dwi_to_t1_matrix']],
            registration_image_file=[['subject_id', 'subject_id', 'T1']],
            fa=[['subject_id', 'subject_id', 'fa']],
            md=[['subject_id', 'subject_id', 'md']],
            roi_file=[['subject_id', 'subject_id', 'rois_dwi']])

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
datasource.inputs.field_template = dict(dwi='data/%s/%s.nii.gz', bvecs='data/%s/%s', bvals='data/%s/%s',
    single_fiber_mask='data/%s/%s_%s.nii.gz', wm_mask='data/%s/%s_%s.nii.gz', termination_mask='data/%s/%s_%s.nii.gz',
    registration_matrix_file='data/%s/%s_%s.mat', registration_image_file='data/%s/%s_%s.nii.gz',
    fa='data/%s/%s_%s.nii.gz', md='data/%s/%s_%s.nii.gz', roi_file='data/%s/%s_%s.nii.gz')
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = True

datasink = pe.Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = op.abspath(name)

dmnwf = create_dmn_pipeline_step2()

# For C.H.U. Liege Siemens Allegra 3T
dmnwf.inputs.tracking.fsl2mrtrix.invert_x = True
dmnwf.inputs.tracking.fsl2mrtrix.invert_y = False
dmnwf.inputs.tracking.fsl2mrtrix.invert_z = False

# Set a number of tracks
dmnwf.inputs.tracking.CSDstreamtrack.desired_number_of_tracks = 10000
dmnwf.inputs.tracking.CSDstreamtrack.minimum_tract_length = 10


workflow = pe.Workflow(name=name)
workflow.base_dir = name

workflow.connect([(infosource, datasource, [('subject_id', 'subject_id')])])
workflow.connect([(infosource, datasink, [('subject_id', '@subject_id')])])

workflow.connect([(infosource, dmnwf, [('subject_id', 'inputnode.subject_id')])])

workflow.connect([(datasource, dmnwf, [('dwi', 'inputnode.dwi')])])
workflow.connect([(datasource, dmnwf, [('bvecs', 'inputnode.bvecs')])])
workflow.connect([(datasource, dmnwf, [('bvals', 'inputnode.bvals')])])
workflow.connect([(datasource, dmnwf, [('single_fiber_mask', 'inputnode.single_fiber_mask')])])
workflow.connect([(datasource, dmnwf, [('wm_mask', 'inputnode.wm_mask')])])
workflow.connect([(datasource, dmnwf, [('termination_mask', 'inputnode.termination_mask')])])
workflow.connect([(datasource, dmnwf, [('registration_matrix_file', 'inputnode.registration_matrix_file')])])
workflow.connect([(datasource, dmnwf, [('registration_image_file', 'inputnode.registration_image_file')])])

workflow.connect([(datasource, dmnwf, [('fa', 'inputnode.fa')])])
workflow.connect([(datasource, dmnwf, [('md', 'inputnode.md')])])
workflow.connect([(datasource, dmnwf, [('roi_file', 'inputnode.roi_file')])])

workflow.connect([(dmnwf, datasink, [("outputnode.fiber_odfs", "subject_id.@fiber_odfs"),
                                   ("outputnode.fiber_tracks_tck_dwi", "subject_id.@fiber_tracks_tck_dwi"),
                                   ("outputnode.fiber_tracks_trk_t1", "subject_id.@fiber_tracks_trk_t1"),
                                   ("outputnode.connectivity_files", "subject_id.@connectivity_files"),
                                   ("outputnode.connectivity_data", "subject_id.@connectivity_data"),
                                   ])])

workflow.config['execution'] = {'remove_unnecessary_outputs': 'false',
                                   'hash_method': 'timestamp'}
workflow.write_graph()
workflow.run()
#workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 3})
