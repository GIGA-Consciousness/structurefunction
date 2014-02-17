import os, os.path as op                      # system functions
import nipype.interfaces.cmtk as cmtk
import nipype.algorithms.misc as misc

def create_rsfmri_correlation_network(name="functional", have_nodes_already=False):
    inputnode_within = pe.Node(interface=util.IdentityInterface(fields=["subject_id", "functional_images", "segmentation_file", "resolution_network_file"]), name="inputnode_within")

    # Create the resampling nodes. Functional images must have the same dimensions as the segmentation file
    resampleFunctional = pe.MapNode(interface=fs.MRIConvert(), name='resampleFunctional', iterfield=['in_file'])
    resampleFunctional.inputs.out_type = 'nii'

    # Create the nodes
    if not have_nodes_already:
        createnodes = pe.Node(interface=cmtk.CreateNodes(), name="CreateNodes")

    # Define the correlation mapping node, the CFF Converter, and NetworkX MATLAB -> CommaSeparatedValue node
    time_course_correlation = pe.Node(interface=cmtk.SimpleTimeCourseCorrelationGraph(), name='time_course_correlation')
    correlationCFFConverter = pe.Node(interface=cmtk.CFFConverter(), name="correlationCFFConverter")
    correlationCFFConverter.inputs.out_file = 'correlation.cff'
    correlationNetworkXMetrics = pe.Node(interface=cmtk.NetworkXMetrics(), name="correlationNetworkXMetrics")
    correlationMatlab2CSV_nx = pe.Node(interface=misc.Matlab2CSV(), name="correlationMatlab2CSV_nx")

    #### Create the workflow ###
    cor_ntwk = pe.Workflow(name='cor_ntwk')

    # Calculates the the t-value threshold for each node/IC
    cor_ntwk.connect([(inputnode_within, resampleFunctional,[('functional_images', 'in_file')])])
    cor_ntwk.connect([(inputnode_within, resampleFunctional,[('segmentation_file', 'reslice_like')])])
    cor_ntwk.connect([(resampleFunctional, time_course_correlation,[('out_file', 'in_files')])])

    # Creates the nodes for the graph from the input segmentation file and resolution network file
    if not have_nodes_already:
        cor_ntwk.connect([(inputnode_within, createnodes,[('segmentation_file', 'roi_file')])])
        cor_ntwk.connect([(inputnode_within, createnodes,[('resolution_network_file', 'resolution_network_file')])])
        cor_ntwk.connect([(createnodes, time_course_correlation,[('node_network', 'structural_network')])])
    else:
        cor_ntwk.connect([(inputnode_within, time_course_correlation,[('resolution_network_file', 'structural_network')])])

    # Creates a connectivity graph for each IC and stores all of the graphs in a CFF file
    cor_ntwk.connect([(inputnode_within, time_course_correlation,[('segmentation_file', 'segmentation_file')])])
    cor_ntwk.connect([(time_course_correlation, correlationCFFConverter,[('network_file', 'gpickled_networks')])])
    cor_ntwk.connect([(time_course_correlation, correlationNetworkXMetrics,[('network_file', 'in_file')])])
    cor_ntwk.connect([(correlationNetworkXMetrics, correlationMatlab2CSV_nx,[("node_measures_matlab","in_file")])])

    # Create a higher-level workflow
    inputnode = pe.Node(interface=util.IdentityInterface(fields=["subject_id", "functional_images", "segmentation_file", "resolution_network_file"]), name="inputnode")

    outputnode = pe.Node(interface = util.IdentityInterface(fields=["correlation_ntwk", "correlation_cff"]), name="outputnode")

    correlation = pe.Workflow(name=name)
    correlation.base_output_dir=name
    correlation.base_dir=name
    correlation.connect([(inputnode, cor_ntwk, [("subject_id", "inputnode_within.subject_id"),
                                              ("functional_images", "inputnode_within.functional_images"),
                                              ("segmentation_file", "inputnode_within.segmentation_file"),
                                              ("resolution_network_file", "inputnode_within.resolution_network_file")])
                                              ])

    correlation.connect([(cor_ntwk, outputnode,[('time_course_correlation.network_file', 'correlation_ntwk')])])
    correlation.connect([(cor_ntwk, outputnode,[('correlationCFFConverter.connectome_file', 'correlation_cff')])])
    return correlation
