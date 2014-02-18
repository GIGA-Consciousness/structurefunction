import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import os.path as op
import nipype.interfaces.cmtk as cmtk
import nipype.interfaces.freesurfer as fs
import nipype.pipeline.engine as pe          # pypeline engine
import coma.interfaces as ci

from coma.datasets import sample
data_path = sample.data_path()

subjects_dir = op.join(data_path,"subjects")
output_dir = op.abspath('fdgpet')

info = dict(fdg_pet_image=[['subject_id', '*fswp*']],)

subject_list = ['Bend1']

infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                     name="infosource")
infosource.iterables = ('subject_id', subject_list)
datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=info.keys()),
                     name='datasource')

datasource.inputs.template = "%s/%s"
datasource.inputs.base_directory = data_path
datasource.inputs.field_template = dict(fdg_pet_image='data/%s/.nii')
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = True

parcellate = pe.Node(interface=cmtk.Parcellate(), name="parcellate")
parcellate.inputs.subjects_dir = subjects_dir

resample_fdg_pet = pe.Node(interface=fs.MRIConvert(), name='resample_fdg_pet')
resample_fdg_pet.inputs.out_type = 'nii'

regional_metabolism = pe.Node(interface=ci.RegionalValues(), name="regional_metabolism")
regional_metabolism.inputs.out_stats_file = 'regional_fdg_uptake.mat'

datasink = pe.Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir

workflow = pe.Workflow(name='pet')
workflow.base_dir = output_dir

workflow.connect([(infosource, datasource, [('subject_id', 'subject_id')])])
workflow.connect([(infosource, parcellate, [('subject_id', 'subject_id')])])
workflow.connect([(infosource, datasink, [('subject_id', '@subject_id')])])

workflow.connect([(datasource, resample_fdg_pet, [('fdg_pet_image', 'in_file')])])
workflow.connect([(parcellate, resample_fdg_pet, [('roi_file', 'reslice_like')])])
workflow.connect(
    [(parcellate, regional_metabolism, [('roi_file', 'segmentation_file')])])
workflow.connect(
    [(resample_fdg_pet, regional_metabolism, [('out_file', 'in_files')])])

workflow.connect(
    [(regional_metabolism, datasink, [('networks', '@subject_id.regional_metabolism')])])
workflow.connect(
    [(regional_metabolism, datasink, [('stats_file', '@subject_id.regional_metabolic_stats')])])

workflow.run()
#workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 3})
