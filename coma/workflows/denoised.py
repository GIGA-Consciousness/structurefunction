import os, os.path as op                      # system functions
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.interfaces.cmtk as cmtk
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs
import nipype.pipeline.engine as pe
from nipype.workflows.dmri.connectivity.group_connectivity import (pullnodeIDs)

import coma

def create_denoised_timecourse_workflow(name="pet"):
    try: 
        coma_rest_lib_path = os.environ['COMA_REST_LIB_ROOT']
    except KeyError:
        print 'COMA_REST_LIB_ROOT environment variable not set.'

    inputnode_within = pe.Node(interface=util.IdentityInterface(fields=["subject_id", "functional_images", "fmri_ICA_maps", "ica_mask_image", "fmri_ICA_timecourse", "segmentation_file", "repetition_time", "resolution_network_file"]), name="inputnode_within")

    # Create the resampling nodes. Functional images and ICA maps must have the same dimensions as the segmentation file
    resampleFunctional = pe.MapNode(interface=fs.MRIConvert(), name='resampleFunctional', iterfield=['in_file'])
    resampleFunctional.inputs.out_type = 'nii'
    resampleICAmaps = pe.MapNode(interface=fs.MRIConvert(), name='resampleICAmaps', iterfield=['in_file'])
    resampleICAmaps.inputs.out_type = 'nii'
    resample_neuronal = pe.MapNode(interface=fs.MRIConvert(), name='resample_neuronal', iterfield=['in_file'])
    resample_neuronal.inputs.out_type = 'nii'

    # Create the ComaRestLib nodes
    denoised_image = pe.Node(interface=coma.CreateDenoisedImage(), name='denoised_image')
    denoised_image.inputs.coma_rest_lib_path = coma_rest_lib_path
    matching_classification = pe.Node(interface=coma.MatchingClassification(), name='matching_classification')
    matching_classification.inputs.coma_rest_lib_path = coma_rest_lib_path
    compute_fingerprints = pe.MapNode(interface=coma.ComputeFingerprint(), name='compute_fingerprints', iterfield=['in_file', 'component_index'])
    compute_fingerprints.inputs.coma_rest_lib_path = coma_rest_lib_path

    # Create the functional connectivity thresholding and mapping nodes
    createnodes = pe.Node(interface=cmtk.CreateNodes(), name="CreateNodes")
    neuronal_regional_timecourses = pe.Node(interface=coma.RegionalValues(), name="neuronal_regional_timecourses")
    simple_regional_timecourses = pe.Node(interface=coma.RegionalValues(), name="simple_regional_timecourses")

    split_neuronal = pe.Node(interface=fsl.Split(), name='split_neuronal')
    split_neuronal.inputs.dimension = 't'

    ### Create the workflow ###
    func_ntwk = pe.Workflow(name='func_ntwk')

    # Create the denoised image
    func_ntwk.connect([(inputnode_within, denoised_image,[('fmri_ICA_maps', 'in_files')])])
    func_ntwk.connect([(inputnode_within, denoised_image,[('repetition_time', 'repetition_time')])])    
    func_ntwk.connect([(inputnode_within, denoised_image,[('ica_mask_image', 'ica_mask_image')])])
    func_ntwk.connect([(inputnode_within, denoised_image,[('fmri_ICA_timecourse', 'time_course_image')])])

    # Runs the matching classification
    func_ntwk.connect([(inputnode_within, matching_classification,[('fmri_ICA_maps', 'in_files')])])
    func_ntwk.connect([(inputnode_within, matching_classification,[('repetition_time', 'repetition_time')])])
    func_ntwk.connect([(inputnode_within, matching_classification,[('ica_mask_image', 'ica_mask_image')])])
    func_ntwk.connect([(inputnode_within, matching_classification,[('fmri_ICA_timecourse', 'time_course_image')])])

    # Computes and saves the fingerprint for each IC
    func_ntwk.connect([(inputnode_within, compute_fingerprints,[('fmri_ICA_maps', 'in_file')])])
    func_ntwk.connect([(inputnode_within, compute_fingerprints,[('repetition_time', 'repetition_time')])])
    func_ntwk.connect([(inputnode_within, compute_fingerprints,[('ica_mask_image', 'ica_mask_image')])])
    func_ntwk.connect([(inputnode_within, compute_fingerprints,[('fmri_ICA_timecourse', 'time_course_image')])])
    func_ntwk.connect([(inputnode_within, compute_fingerprints,[(('fmri_ICA_maps', get_component_index), 'component_index')])])

    # Calculates the the t-value threshold for each node/IC
    func_ntwk.connect([(inputnode_within, resampleFunctional,[('functional_images', 'in_file')])])
    func_ntwk.connect([(inputnode_within, resampleFunctional,[('segmentation_file', 'reslice_like')])])

    # Resamples the ICA z-score maps to the same dimensions as the segmentation file
    func_ntwk.connect([(inputnode_within, resampleICAmaps,[('fmri_ICA_maps', 'in_file')])])
    func_ntwk.connect([(inputnode_within, resampleICAmaps,[('segmentation_file', 'reslice_like')])])

    # Splits the 4d neuronal and non-neuronal images, resamples them, and creates the time-course correlation graph
    func_ntwk.connect([(denoised_image, split_neuronal,[('neuronal_image', 'in_file')])])
    func_ntwk.connect([(split_neuronal, resample_neuronal,[('out_files', 'in_file')])])
    func_ntwk.connect([(inputnode_within, resample_neuronal,[('segmentation_file', 'reslice_like')])])

    # Calculates the fmri timecourse for the preprocessed fMRI signal
    func_ntwk.connect([(inputnode_within, simple_regional_timecourses,[('segmentation_file', 'segmentation_file')])])
    func_ntwk.connect([(createnodes, simple_regional_timecourses,[('node_network', 'resolution_network_file')])])
    func_ntwk.connect([(resampleFunctional, simple_regional_timecourses,[('out_file', 'in_files')])])
    simple_regional_timecourses.inputs.out_stats_file = 'fmri_timecourses.mat'

    # Calculates the fmri timecourse for the denoised fMRI signal
    func_ntwk.connect([(inputnode_within, neuronal_regional_timecourses,[('segmentation_file', 'segmentation_file')])])
    func_ntwk.connect([(createnodes, neuronal_regional_timecourses,[('node_network', 'resolution_network_file')])])
    func_ntwk.connect([(resample_neuronal, neuronal_regional_timecourses,[('out_file', 'in_files')])])
    neuronal_regional_timecourses.inputs.out_stats_file = 'denoised_fmri_timecourse.mat'

    # Create a higher-level workflow
    inputnode = pe.Node(interface=util.IdentityInterface(fields=["subject_id", "functional_images", "fmri_ICA_maps", "ica_mask_image", "fmri_ICA_timecourse", "segmentation_file", "repetition_time", "resolution_network_file"]), name="inputnode")

    outputnode = pe.Node(interface = util.IdentityInterface(fields=["simple_regional_timecourse_stats", "neuronal_regional_timecourse_stats"]), name="outputnode")

    functional = pe.Workflow(name=name)
    functional.base_output_dir=name
    functional.base_dir=name
    functional.connect([(inputnode, func_ntwk, [("subject_id", "inputnode_within.subject_id"),
                                              ("functional_images", "inputnode_within.functional_images"),
                                              ("fmri_ICA_timecourse", "inputnode_within.fmri_ICA_timecourse"),
                                              ("fmri_ICA_maps", "inputnode_within.fmri_ICA_maps"),
                                              ("ica_mask_image", "inputnode_within.ica_mask_image"),
                                              ("segmentation_file", "inputnode_within.segmentation_file"),
                                              ("repetition_time", "inputnode_within.repetition_time"),
                                              ("resolution_network_file", "inputnode_within.resolution_network_file")])
                                              ])

    functional.connect([(func_ntwk, outputnode,[('grouped_graphs.out_files', 'neuronal_ntwks')])])
    functional.connect([(func_ntwk, outputnode,[('neuronalCFFConverter.connectome_file', 'neuronal_cff')])])
    functional.connect([(func_ntwk, outputnode,[('neuronal_regional_timecourses.stats_file', 'neuronal_regional_timecourse_stats')])])
    
    functional.connect([(func_ntwk, outputnode,[('simple_regional_timecourses.stats_file', 'simple_regional_timecourse_stats')])])

    functional.connect([(func_ntwk, outputnode,[('remove_unconnected_corr.out_files', 'correlation_ntwks')])])
    functional.connect([(func_ntwk, outputnode,[('correlationCFFConverter.connectome_file', 'correlation_cff')])])
    functional.connect([(func_ntwk, outputnode,[('add_subjid_to_csv_cor.csv_file', 'correlation_stats')])])

    functional.connect([(func_ntwk, outputnode,[('remove_unconnected_anticorr.out_files', 'anticorrelation_ntwks')])])
    functional.connect([(func_ntwk, outputnode,[('anticorrelationCFFConverter.connectome_file', 'anticorrelation_cff')])])
    functional.connect([(func_ntwk, outputnode,[('add_subjid_to_csv_anticor.csv_file', 'anticorrelation_stats')])])

    functional.connect([(func_ntwk, outputnode,[('matching_classification.stats_file', 'matching_stats')])])

    return functional