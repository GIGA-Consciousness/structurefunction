structurefunction
=================

The goal of this repository is to make it easier to use some of the RestLib functions in connected pipelines.
Other methods, such as fMRI IC graph analysis, simple fMRI timecourse correlation analysis, and PET regional value (e.g. mean, max, min) extraction, are also included. Some examples may demonstrate how to use these methods with other implemented workflows, such as the diffusion tractography workflows in Nipype.

Installation
------------
1. First, clone the repository and install the python package

        git clone https://github.com/swederik/structurefunction.git
        cd structurefunction
        python setup.py install

2. Next, make sure you have GIFT ICA (if you want to run ICA) and the Rest Library in your MATLAB path.
    * Download it from here: http://mialab.mrn.org/software/gift/index.html
    * Make sure to use "Add with Subfolders" in MATLAB.

3. Add two environment variables pointing to the Rest Library and the structurefunction repository:

        export COMA_REST_LIB_ROOT=/path/to/structurefunction/RestLib
        export COMA_DIR=/path/to/structurefunction
    
    if you are using bash or zsh, add them to your ( ~/.bashrc or  ~/.zshrc). For csh (~/.cshrc), add:
    
        setenv COMA_REST_LIB_ROOT /path/to/structurefunction/RestLib
        setenv COMA_DIR /path/to/structurefunction
        
   These should not actually say /path/to, but should describe the path ON YOUR COMPUTER to the files in question. E.g. it could be something like /home/erik/structurefunction on Linux or /Users/erik/structurefunction on a Mac. 

4. Make sure the RestLib is in your MATLAB path.

5. OPTIONAL: If RestLib is not checked out and the folder is empty, you may need to run:

        git submodule update --init --recursive

Python Dependencies
------------

1. MNE-Python (https://github.com/mne-tools/mne-python, just for file fetching utils)
    
        pip install mne

2. ConnectomeMapper (https://github.com/LTS5/cmp)

        git clone https://github.com/LTS5/cmp.git
        cd cmp
        git checkout 78608e986634341c1a0cb08ed0b7ea8a632307e3
        python setup.py install


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
    
    
