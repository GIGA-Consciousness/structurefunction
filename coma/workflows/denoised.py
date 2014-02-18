import os
import os.path as op                      # system functions
import nipype.interfaces.utility as util     # utility
import nipype.interfaces.cmtk as cmtk
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs
import nipype.pipeline.engine as pe
import coma.interfaces as ci

from ..interfaces import SingleSubjectICA, CreateDenoisedImage, MatchingClassification, ComputeFingerprint
from ..helpers import get_component_index

def create_denoised_timecourse_workflow(name="denoised"):
    try: 
        coma_rest_lib_path = os.environ['COMA_REST_LIB_ROOT']
    except KeyError:
        print 'COMA_REST_LIB_ROOT environment variable not set.'

    inputnode = pe.Node(interface=util.IdentityInterface(fields=["subject_id", "functional_images",
        "segmentation_file", "repetition_time"]), name="inputnode")
    outputnode = pe.Node(interface = util.IdentityInterface(fields=["stats_file"]), name="outputnode")

    # Create the GIFT ICA node
    ica = pe.Node(interface=SingleSubjectICA(), name='ica')

    # Create the resampling nodes. Functional images and ICA maps must have the same dimensions as the segmentation file
    resampleFunctional = pe.MapNode(interface=fs.MRIConvert(), name='resampleFunctional', iterfield=['in_file'])
    resampleFunctional.inputs.out_type = 'nii'
    
    resampleICAmaps = pe.MapNode(interface=fs.MRIConvert(), name='resampleICAmaps', iterfield=['in_file'])
    resampleICAmaps.inputs.out_type = 'nii'
    
    resample_neuronal = pe.MapNode(interface=fs.MRIConvert(), name='resample_neuronal', iterfield=['in_file'])
    resample_neuronal.inputs.out_type = 'nii'

    # Create the ComaRestLib nodes
    denoised_image = pe.Node(interface=CreateDenoisedImage(), name='denoised_image')
    denoised_image.inputs.coma_rest_lib_path = coma_rest_lib_path
    
    matching_classification = pe.Node(interface=MatchingClassification(), name='matching_classification')
    matching_classification.inputs.coma_rest_lib_path = coma_rest_lib_path

    compute_fingerprints = pe.MapNode(interface=ComputeFingerprint(), name='compute_fingerprints', iterfield=['in_file', 'component_index'])
    compute_fingerprints.inputs.coma_rest_lib_path = coma_rest_lib_path

    # Create nodes for ConnectomeViewer and calculate the neuronal timecourses
    #createnodes = pe.Node(interface=cmtk.CreateNodes(), name="CreateNodes")

    split_ICs = pe.Node(interface=fsl.Split(), name='split_ICs')
    split_ICs.inputs.dimension = 't'

    split_neuronal = split_ICs.clone("split_neuronal")

    neuronal_regional_timecourses = pe.Node(interface=ci.RegionalValues(), name="neuronal_regional_timecourses")
    neuronal_regional_timecourses.inputs.out_stats_file = 'denoised_fmri_timecourse.mat'

    # --------- Create the workflow ---------#
    workflow = pe.Workflow(name=name)

    # Run ICA with GIFT
    workflow.connect([(inputnode, ica,[('functional_images', 'in_files')])])
    workflow.connect([(inputnode, ica,[('subject_id', 'prefix')])])

    # Create Nodes from segmentation file
    #workflow.connect([(inputnode, createnodes,[('segmentation_file', 'roi_file')])])

    # Create the denoised image
    workflow.connect([(inputnode, denoised_image,[('repetition_time', 'repetition_time')])])
    workflow.connect([(ica, split_ICs,[('independent_component_images', 'in_file')])])
    workflow.connect([(split_ICs, denoised_image,[('out_files', 'in_files')])])
    workflow.connect([(ica, denoised_image,[('mask_image', 'ica_mask_image')])])
    workflow.connect([(ica, denoised_image,[('independent_component_timecourse', 'time_course_image')])])

    # Runs the matching classification
    workflow.connect([(inputnode, matching_classification,[('repetition_time', 'repetition_time')])])
    workflow.connect([(split_ICs, matching_classification,[('out_files', 'in_files')])])
    workflow.connect([(ica, matching_classification,[('mask_image', 'ica_mask_image')])])
    workflow.connect([(ica, matching_classification,[('independent_component_timecourse', 'time_course_image')])])

    # Computes and saves the fingerprint for each IC
    workflow.connect([(inputnode, compute_fingerprints,[('repetition_time', 'repetition_time')])])
    workflow.connect([(split_ICs, compute_fingerprints,[('out_files', 'in_file')])])
    workflow.connect([(ica, compute_fingerprints,[('mask_image', 'ica_mask_image')])])
    workflow.connect([(ica, compute_fingerprints,[('independent_component_timecourse', 'time_course_image')])])
    workflow.connect([(ica, compute_fingerprints,[(('independent_component_images', get_component_index), 'component_index')])])

    # Resamples the ICA z-score maps to the same dimensions as the segmentation file
    workflow.connect([(ica, resampleICAmaps,[('independent_component_images', 'in_file')])])
    workflow.connect([(inputnode, resampleICAmaps,[('segmentation_file', 'reslice_like')])])

    # Splits the 4d neuronal and non-neuronal images, resamples them, and creates the time-course correlation graph
    workflow.connect([(denoised_image, split_neuronal,[('neuronal_image', 'in_file')])])
    workflow.connect([(split_neuronal, resample_neuronal,[('out_files', 'in_file')])])
    workflow.connect([(inputnode, resample_neuronal,[('segmentation_file', 'reslice_like')])])

    # Calculates the fmri timecourse
    workflow.connect([(inputnode, neuronal_regional_timecourses,[('segmentation_file', 'segmentation_file')])])
    workflow.connect([(resample_neuronal, neuronal_regional_timecourses,[('out_file', 'in_files')])])
    #workflow.connect([(createnodes, neuronal_regional_timecourses,[('node_network', 'resolution_network_file')])])

    # Send stats to outputnode
    workflow.connect([(neuronal_regional_timecourses, outputnode,[('stats_file', 'stats_file')])])
    return workflow