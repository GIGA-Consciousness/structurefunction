import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import os, os.path as op
import nipype.algorithms.misc as misc
import nipype.interfaces.cmtk as cmtk
import nipype.interfaces.freesurfer as fs
import nipype.pipeline.engine as pe          # pypeline engine
import coma.interfaces as ci

from coma.datasets import sample
data_path = sample.data_path()

subjects_dir = op.join(data_path,"subjects")
output_dir = op.abspath('fmri_timecourses')

info = dict(functional_images=[['subject_id', '*swvmsrf*']],
            segmentation_file=[['subject_id','*ROI_scale500*']])

subject_list = ['Bend1']

infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                     name="infosource")
infosource.iterables = ('subject_id', subject_list)
datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=info.keys()),
                     name = 'datasource')

datasource.inputs.template = "%s/%s"
datasource.inputs.base_directory = data_path
datasource.inputs.field_template = dict(functional_images='data/%s/restMotionProcessed/%s.img', segmentation_file='data/%s/%s.nii.gz')
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = True

resample_functional = pe.MapNode(interface=fs.MRIConvert(), name='resample_functional', iterfield=["in_file"])
resample_functional.inputs.out_type = 'nii'

fmri_timecourses = pe.Node(interface=ci.RegionalValues(), name="fmri_timecourses")
fmri_timecourses.inputs.out_stats_file = 'fmri_timecourses.mat'

datasink = pe.Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.raise_on_empty = False
datasink.inputs.ignore_exception = True

workflow = pe.Workflow(name='timecourses')
workflow.base_dir = output_dir

workflow.connect([(infosource, datasource,[('subject_id', 'subject_id')])])
workflow.connect([(datasource, fmri_timecourses,[('segmentation_file','segmentation_file')])])
workflow.connect([(datasource, resample_functional,[('functional_images','in_file')])])
workflow.connect([(datasource, resample_functional,[('segmentation_file','reslice_like')])])
workflow.connect([(resample_functional, fmri_timecourses,[('out_file','in_files')])])
workflow.connect([(fmri_timecourses, datasink,[('stats_file','@subject_id.fmri_timecourses')])])

workflow.run()
