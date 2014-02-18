structurefunction
=================

The goal of this repository is to make it easier to use some of the RestLib functions in connected pipelines.
Other methods, such as fMRI IC graph analysis, simple fMRI timecourse correlation analysis, and PET regional value (e.g. mean, max, min) extraction, are also included. Some examples may demonstrate how to use these methods with other implemented workflows, such as the diffusion tractography workflows in Nipype.

Installation
------------
First, clone the repository and install the python package

    git clone https://github.com/swederik/structurefunction.git
    cd structurefunction
    python setup.py install

Next, make sure you have GIFT ICA (if you want to run ICA) and the Rest Library in your MATLAB path.
Download it from here: http://mialab.mrn.org/software/gift/index.html


Finally, add two environment variables pointing to the Rest Library and the structurefunction repository:

    export COMA_REST_LIB_ROOT=/path/to/structurefunction/RestLib
    export COMA_DIR=/path/to/structurefunction
    
if you are using bash or zsh.

Interface Example
-----------------

Use interfaces as you would Nipype interfaces.
For example, to run Independent Component Analysis with GIFT, you would use:

    import coma.interfaces as ci
    ica = ci.SingleSubjectICA()
    ica.inputs.in_files = ["f1.nii", "f2.nii"]
    ica.inputs.desired_number_of_components = 30
    ica.inputs.prefix = "Erik"
    ica.run()
    
Workflow Example
-----------------

    import coma.workflows as cw
    denoised = cw.create_denoised_timecourse_workflow()
    
    
