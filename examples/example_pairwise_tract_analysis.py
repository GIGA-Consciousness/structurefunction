import os
import os.path as op
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from coma.workflows.dmn import create_paired_tract_analysis_wf

paired = create_paired_tract_analysis_wf("example_paired")
paired


paired.write_graph()
paired.run()
#workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 4})