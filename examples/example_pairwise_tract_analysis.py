import os
import os.path as op
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from coma.workflows.dmn import create_paired_tract_analysis_wf

path = op.abspath("/Users/erik/Dropbox/Thesis/Data/Bend1")

paired = create_paired_tract_analysis_wf("example_paired")
paired.base_dir = op.abspath("example_paired")
paired.inputs.inputnode.fa = op.join(path,"Bend1_fa.nii.gz")
paired.inputs.inputnode.md = op.join(path,"Bend1_md.nii.gz")
paired.inputs.inputnode.roi_file = op.join(path,"DMN_ROIs_DTIReg.nii")
paired.inputs.inputnode.track_file = op.join(path,"Bend1_tracks_50k.tck")
paired.write_graph()
paired.config['execution'] = {'remove_unnecessary_outputs': 'false',
                                   'hash_method': 'timestamp'}
paired.run()
#workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 4})