import os, os.path as op                      # system functions
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.interfaces.cmtk as cmtk
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs
import nipype.algorithms.misc as misc
import nipype.pipeline.engine as pe
from ..interfaces import SingleSubjectICA, MatchingClassification, ComputeFingerprint, CreateDenoisedImage
from nipype.interfaces.utility import Function
from nipype.workflows.dmri.connectivity.group_connectivity import (pullnodeIDs, concatcsv)

def group_fmri_graphs(subject_id, in_file, component_index, matching_stats):
    def flatten_arrays(array_of_arrays):
        list_of_lists = array_of_arrays.tolist()
        result = [];
        map(result.extend, list_of_lists)
        return result
    import scipy.io as sio
    import shutil
    from nipype.utils.filemanip import split_filename
    import os, os.path as op
    stats = sio.loadmat(matching_stats)
    templates = flatten_arrays(stats['templates'])
    template_names = flatten_arrays(stats['namesTemplate'][0])
    components = flatten_arrays(stats['components'])
    gofs = flatten_arrays(stats['gofs'])
    neuronal_bool = flatten_arrays(stats['neuronal_bool'])
    neuronal_prob = flatten_arrays(stats['neuronal_prob'])
    path, name, ext = split_filename(in_file)
    out_file = in_file
    try:
        IC_idx = components.index(int(component_index))
        if int(neuronal_bool[IC_idx]) == 1:
            template_name = str(template_names[templates[IC_idx]-1]) # Subtract one because template index comes from MATLAB
            new_name = str(subject_id) + '_' + template_name + '_IC' + str(component_index)
            out_file = op.abspath(new_name + ext)
            shutil.copyfile(in_file, out_file)
            print 'Component {ic} is neuronal with probability {p}. Assigning to template {t}'.format(ic=component_index, p=neuronal_prob[IC_idx], t=template_name)
            return out_file
    except ValueError:
        return None


def removeNoneValues(in_files):
    out_files = []
    for value in in_files:
        if not value is None:
            out_files.append(value)
    return out_files


def create_fmri_graphs(name="functional", with_simple_timecourse_correlation=False):
    try:
        coma_rest_lib_path = os.environ['COMA_REST_LIB_ROOT']
    except KeyError:
        print 'COMA_REST_LIB_ROOT environment variable not set.'

    inputnode_within = pe.Node(interface=util.IdentityInterface(fields=["subject_id", "functional_images", "segmentation_file", "repetition_time", "resolution_network_file"]), name="inputnode_within")

    ica = pe.Node(interface=SingleSubjectICA(), name='ica')
    # Create the resampling nodes. Functional images and ICA maps must have the same dimensions as the segmentation file
    resampleFunctional = pe.MapNode(interface=fs.MRIConvert(), name='resampleFunctional', iterfield=['in_file'])
    resampleFunctional.inputs.out_type = 'nii'
    resampleICAmaps = pe.MapNode(interface=fs.MRIConvert(), name='resampleICAmaps', iterfield=['in_file'])
    resampleICAmaps.inputs.out_type = 'nii'
    resample_neuronal = pe.MapNode(interface=fs.MRIConvert(), name='resample_neuronal', iterfield=['in_file'])
    resample_neuronal.inputs.out_type = 'nii'
    if with_simple_timecourse_correlation:
        resample_non_neuronal = pe.MapNode(interface=fs.MRIConvert(), name='resample_non_neuronal', iterfield=['in_file'])
        resample_non_neuronal.inputs.out_type = 'nii'

    # Create the ComaRestLib nodes
    denoised_image = pe.Node(interface=CreateDenoisedImage(), name='denoised_image')
    denoised_image.inputs.coma_rest_lib_path = coma_rest_lib_path
    matching_classification = pe.Node(interface=MatchingClassification(), name='matching_classification')
    matching_classification.inputs.coma_rest_lib_path = coma_rest_lib_path
    compute_fingerprints = pe.MapNode(interface=ComputeFingerprint(), name='compute_fingerprints', iterfield=['in_file', 'component_index'])
    compute_fingerprints.inputs.coma_rest_lib_path = coma_rest_lib_path

    # Create the functional connectivity thresholding and mapping nodes
    createnodes = pe.Node(interface=cmtk.CreateNodes(), name="CreateNodes")
    connectivity_threshold = pe.Node(interface=cmtk.CreateConnectivityThreshold(), name='connectivity_threshold')
    connectivity_graph = pe.MapNode(interface=cmtk.ConnectivityGraph(), name='connectivity_graph', iterfield=['in_file', 'component_index'])
    neuronal_regional_timecourses = pe.Node(interface=cmtk.RegionalValues(), name="neuronal_regional_timecourses")

    # Define the CFF Converter, NetworkX MATLAB -> CommaSeparatedValue nodes
    graphCFFConverter = pe.Node(interface=cmtk.CFFConverter(), name="graphCFFConverter")
    neuronalCFFConverter = pe.Node(interface=cmtk.CFFConverter(), name="neuronalCFFConverter")
    neuronalCFFConverter.inputs.out_file = 'neuronal.cff'

    correlationCFFConverter = pe.Node(interface=cmtk.CFFConverter(), name="correlationCFFConverter")
    correlationCFFConverter.inputs.out_file = 'correlation.cff'
    ConnectivityGraphNetworkXMetrics_correlation = pe.MapNode(interface=cmtk.NetworkXMetrics(), name="cor_fMRIConnectivityGraphNetworkXMetrics", iterfield=['in_file'])
    Matlab2CSV_nx_cor = pe.MapNode(interface=misc.Matlab2CSV(), name="Matlab2CSV_nx_cor", iterfield=['in_file'])

    anticorrelationCFFConverter = pe.Node(interface=cmtk.CFFConverter(), name="anticorrelationCFFConverter")
    anticorrelationCFFConverter.inputs.out_file = 'anticorrelation.cff'
    ConnectivityGraphNetworkXMetrics_anticorrelation = pe.MapNode(interface=cmtk.NetworkXMetrics(), name="anticor_fMRIConnectivityGraphNetworkXMetrics", iterfield=['in_file'])
    Matlab2CSV_nx_anticor = pe.MapNode(interface=misc.Matlab2CSV(), name="Matlab2CSV_nx_anticor", iterfield=['in_file'])

    MergeCSVFiles_cor = pe.MapNode(interface=misc.MergeCSVFiles(), name="MergeCSVFiles_cor", iterfield=['in_files', 'extra_field'])
    MergeCSVFiles_cor.inputs.extra_column_heading = 'template_name'
    MergeCSVFiles_anticor = MergeCSVFiles_cor.clone('MergeCSVFiles_anticor')

    # Uses a Function interface to group the fMRI graphs and save the neuronal graphs with more detailed names
    group_fmri_graphs_interface = Function(input_names=["subject_id", "in_file", "component_index", "matching_stats"],
                             output_names=["out_file"],
                             function=group_fmri_graphs)

    # Create lists of the neuronal, correlation, and anticorrelation graphs
    group_graphs = pe.MapNode(interface=group_fmri_graphs_interface, name='group_graphs', iterfield=['in_file', 'component_index'])
    group_graphs_corr = pe.MapNode(interface=group_fmri_graphs_interface, name='group_graphs_corr', iterfield=['in_file', 'component_index'])
    group_graphs_anticorr = pe.MapNode(interface=group_fmri_graphs_interface, name='group_graphs_anticorr', iterfield=['in_file', 'component_index'])

    # Define a simple interface for a function which removes 'None' values from the graph grouping above
    removeNoneValues_interface = Function(input_names=["in_files"],
                             output_names=["out_files"],
                             function=removeNoneValues)

    # Create nodes for the removing 'None' values from the neuronal, correlation, and anticorrelation lists
    grouped_graphs = pe.Node(interface=removeNoneValues_interface, name='grouped_graphs')
    grouped_graphs_corr = pe.Node(interface=removeNoneValues_interface, name='grouped_graphs_corr')
    grouped_graphs_anticorr = pe.Node(interface=removeNoneValues_interface, name='grouped_graphs_anticorr')

    remove_unconnected_graphs_interface = Function(input_names=["in_files"],
                             output_names=["out_files"],
                             function=remove_unconnected_graphs)

    remove_unconnected_corr = pe.Node(interface=remove_unconnected_graphs_interface, name='remove_unconnected_corr')
    remove_unconnected_anticorr = remove_unconnected_corr.clone(name='remove_unconnected_anticorr')

    concat_csv_interface = Function(input_names=["in_files"], output_names=["out_name"],
                             function=concatcsv)

    concatcsv_cor = pe.Node(interface=concat_csv_interface, name='concatcsv_cor')
    concatcsv_anticor = pe.Node(interface=concat_csv_interface, name='concatcsv_anticor')
    add_subjid_to_csv_cor = pe.Node(interface=misc.AddCSVColumn(), name='add_subjid_to_csv_cor')
    add_subjid_to_csv_cor.inputs.extra_column_heading = 'Subject'

    add_subjid_to_csv_anticor = pe.Node(interface=misc.AddCSVColumn(), name='add_subjid_to_csv_anticor')
    add_subjid_to_csv_anticor.inputs.extra_column_heading = 'Subject'

    split_neuronal = pe.Node(interface=fsl.Split(), name='split_neuronal')
    split_neuronal.inputs.dimension = 't'
    TCcorrCFFConverter = pe.Node(interface=cmtk.CFFConverter(), name="TCcorrCFFConverter")
    if with_simple_timecourse_correlation:
        split_non_neuronal = split_neuronal.clone(name='split_non_neuronal')
        neuronal_time_course_correlation = pe.Node(interface=cmtk.SimpleTimeCourseCorrelationGraph(), name='neuronal_time_course_correlation')
        neuronal_time_course_correlation.inputs.out_network_file = 'neuronal.pck'
        non_neuronal_time_course_correlation = pe.Node(interface=cmtk.SimpleTimeCourseCorrelationGraph(), name='non_neuronal_time_course_correlation')
        non_neuronal_time_course_correlation.inputs.out_network_file = 'non_neuronal.pck'
        TCcorrCFFConverter.inputs.out_file = 'time_course_correlation.cff'
        mergeSTCC = pe.Node(interface=util.Merge(2), name='mergeSTCC')
    else:
        TCcorrCFFConverter.inputs.out_file = 'time_courses.cff'

    ### Create the workflow ###
    func_ntwk = pe.Workflow(name='func_ntwk')


    func_ntwk.connect([(inputnode_within, ica,[('functional_images', 'in_files')])])
    func_ntwk.connect([(inputnode_within, ica,[('subject_id', 'prefix')])])

    # Create the denoised image
    func_ntwk.connect([(inputnode_within, denoised_image,[('subject_id', 'prefix')])])
    func_ntwk.connect([(inputnode_within, denoised_image,[('repetition_time', 'repetition_time')])])
    func_ntwk.connect([(ica, denoised_image,[('independent_component_images', 'in_files')])])
    func_ntwk.connect([(ica, denoised_image,[('mask_image', 'ica_mask_image')])])
    func_ntwk.connect([(ica, denoised_image,[('independent_component_timecourse', 'time_course_image')])])

    # Runs the matching classification
    func_ntwk.connect([(inputnode_within, matching_classification,[('subject_id', 'prefix')])])
    func_ntwk.connect([(inputnode_within, matching_classification,[('repetition_time', 'repetition_time')])])
    func_ntwk.connect([(ica, matching_classification,[('independent_component_images', 'in_files')])])
    func_ntwk.connect([(ica, matching_classification,[('mask_image', 'ica_mask_image')])])
    func_ntwk.connect([(ica, matching_classification,[('independent_component_timecourse', 'time_course_image')])])

    # Computes and saves the fingerprint for each IC

    func_ntwk.connect([(inputnode_within, compute_fingerprints,[('repetition_time', 'repetition_time')])])
    func_ntwk.connect([(ica, compute_fingerprints,[('independent_component_images', 'in_file')])])
    func_ntwk.connect([(ica, compute_fingerprints,[('mask_image', 'ica_mask_image')])])
    func_ntwk.connect([(ica, compute_fingerprints,[('independent_component_timecourse', 'time_course_image')])])
    func_ntwk.connect([(ica, compute_fingerprints,[(('independent_component_images', get_component_index), 'component_index')])])

    # Calculates the the t-value threshold for each node/IC
    func_ntwk.connect([(inputnode_within, resampleFunctional,[('functional_images', 'in_file')])])
    func_ntwk.connect([(inputnode_within, resampleFunctional,[('segmentation_file', 'reslice_like')])])
    func_ntwk.connect([(resampleFunctional, connectivity_threshold,[('out_file', 'in_files')])])
    func_ntwk.connect([(ica, connectivity_threshold,[('independent_component_timecourse', 'time_course_file')])])
    func_ntwk.connect([(inputnode_within, connectivity_threshold,[('segmentation_file', 'segmentation_file')])])

    # Resamples the ICA z-score maps to the same dimensions as the segmentation file
    func_ntwk.connect([(ica, resampleICAmaps,[('independent_component_images', 'in_file')])])
    func_ntwk.connect([(inputnode_within, resampleICAmaps,[('segmentation_file', 'reslice_like')])])

    # Splits the 4d neuronal and non-neuronal images, resamples them, and creates the time-course correlation graph
    func_ntwk.connect([(denoised_image, split_neuronal,[('neuronal_image', 'in_file')])])
    func_ntwk.connect([(split_neuronal, resample_neuronal,[('out_files', 'in_file')])])
    func_ntwk.connect([(inputnode_within, resample_neuronal,[('segmentation_file', 'reslice_like')])])

    # Calculates the fmri timecourse
    func_ntwk.connect([(inputnode_within, neuronal_regional_timecourses,[('segmentation_file', 'segmentation_file')])])
    func_ntwk.connect([(createnodes, neuronal_regional_timecourses,[('node_network', 'resolution_network_file')])])
    func_ntwk.connect([(resample_neuronal, neuronal_regional_timecourses,[('out_file', 'in_files')])])
    neuronal_regional_timecourses.inputs.out_stats_file = 'denoised_fmri_timecourse.mat'

    if with_simple_timecourse_correlation:
        func_ntwk.connect([(inputnode_within, neuronal_time_course_correlation,[('segmentation_file', 'segmentation_file')])])
        func_ntwk.connect([(createnodes, neuronal_time_course_correlation,[('node_network', 'structural_network')])])
        func_ntwk.connect([(resample_neuronal, neuronal_time_course_correlation,[('out_file', 'in_files')])])
        func_ntwk.connect([(neuronal_time_course_correlation, mergeSTCC,[('network_file', 'in1')])])

        func_ntwk.connect([(denoised_image, split_non_neuronal,[('non_neuronal_image', 'in_file')])])
        func_ntwk.connect([(split_non_neuronal, resample_non_neuronal,[('out_files', 'in_file')])])
        func_ntwk.connect([(inputnode_within, resample_non_neuronal,[('segmentation_file', 'reslice_like')])])
        func_ntwk.connect([(inputnode_within, non_neuronal_time_course_correlation,[('segmentation_file', 'segmentation_file')])])
        func_ntwk.connect([(createnodes, non_neuronal_time_course_correlation,[('node_network', 'structural_network')])])
        func_ntwk.connect([(resample_non_neuronal, non_neuronal_time_course_correlation,[('out_file', 'in_files')])])
        func_ntwk.connect([(non_neuronal_time_course_correlation, mergeSTCC,[('network_file', 'in2')])])
        func_ntwk.connect([(mergeSTCC, TCcorrCFFConverter,[('out', 'gpickled_networks')])])

    # Creates the nodes for the graph from the input segmentation file and resolution network file
    func_ntwk.connect([(inputnode_within, createnodes,[('segmentation_file', 'roi_file')])])
    func_ntwk.connect([(inputnode_within, createnodes,[('resolution_network_file', 'resolution_network_file')])])
    func_ntwk.connect([(inputnode_within, MergeCSVFiles_cor,[(('resolution_network_file', pullnodeIDs), 'row_headings')])])
    func_ntwk.connect([(inputnode_within, MergeCSVFiles_anticor,[(('resolution_network_file', pullnodeIDs), 'row_headings')])])

    # Creates a connectivity graph for each IC and stores all of the graphs in a CFF file
    func_ntwk.connect([(inputnode_within, connectivity_graph,[('segmentation_file', 'segmentation_file')])])
    func_ntwk.connect([(createnodes, connectivity_graph,[('node_network', 'resolution_network_file')])])
    func_ntwk.connect([(resampleICAmaps, connectivity_graph,[('out_file', 'in_file')])])
    func_ntwk.connect([(resampleICAmaps, connectivity_graph,[(('out_file', get_component_index_resampled), 'component_index')])])
    func_ntwk.connect([(connectivity_threshold, connectivity_graph,[('t_value_threshold_file', 't_value_threshold_file')])])
    func_ntwk.connect([(connectivity_graph, graphCFFConverter,[('network_file', 'gpickled_networks')])])

    # Uses the matching classification to separate the neuronal connectivity graphs
    func_ntwk.connect([(inputnode_within, group_graphs,[('subject_id', 'subject_id')])])
    func_ntwk.connect([(connectivity_graph, group_graphs,[('network_file', 'in_file')])])
    func_ntwk.connect([(resampleICAmaps, group_graphs,[(('out_file', get_component_index_resampled), 'component_index')])])
    func_ntwk.connect([(matching_classification, group_graphs,[('stats_file', 'matching_stats')])])
    func_ntwk.connect([(group_graphs, grouped_graphs,[('out_file', 'in_files')])])
    func_ntwk.connect([(grouped_graphs, neuronalCFFConverter,[('out_files', 'gpickled_networks')])])

    # Groups the correlation graphs as above, calculates NetworkX measures, outputs to a CSV file
    func_ntwk.connect([(inputnode_within, group_graphs_corr,[('subject_id', 'subject_id')])])
    func_ntwk.connect([(connectivity_graph, group_graphs_corr,[('correlation_network', 'in_file')])])
    func_ntwk.connect([(resampleICAmaps, group_graphs_corr,[(('out_file', get_component_index_resampled), 'component_index')])])
    func_ntwk.connect([(matching_classification, group_graphs_corr,[('stats_file', 'matching_stats')])])
    func_ntwk.connect([(group_graphs_corr, grouped_graphs_corr,[('out_file', 'in_files')])])
    func_ntwk.connect([(grouped_graphs_corr, correlationCFFConverter,[('out_files', 'gpickled_networks')])])
    func_ntwk.connect([(grouped_graphs_corr, remove_unconnected_corr,[('out_files', 'in_files')])])
    func_ntwk.connect([(grouped_graphs_corr, ConnectivityGraphNetworkXMetrics_correlation,[('out_files', 'in_file')])])
    func_ntwk.connect([(grouped_graphs_corr, MergeCSVFiles_cor,[(('out_files', pull_template_name), 'extra_field')])])
    func_ntwk.connect([(ConnectivityGraphNetworkXMetrics_correlation, Matlab2CSV_nx_cor,[("node_measures_matlab","in_file")])])
    func_ntwk.connect([(Matlab2CSV_nx_cor, MergeCSVFiles_cor,[("csv_files","in_files")])])
    func_ntwk.connect([(MergeCSVFiles_cor, concatcsv_cor,[("csv_file","in_files")])])
    func_ntwk.connect([(concatcsv_cor, add_subjid_to_csv_cor,[("out_name","in_file")])])
    func_ntwk.connect([(inputnode_within, add_subjid_to_csv_cor,[("subject_id","extra_field")])])
    func_ntwk.connect([(inputnode_within, add_subjid_to_csv_cor,[("subject_id","out_file")])])

    # Groups the anticorrelation graphs as above, calculates NetworkX measures, outputs to a CSV file
    func_ntwk.connect([(inputnode_within, group_graphs_anticorr,[('subject_id', 'subject_id')])])
    func_ntwk.connect([(connectivity_graph, group_graphs_anticorr,[('anticorrelation_network', 'in_file')])])
    func_ntwk.connect([(resampleICAmaps, group_graphs_anticorr,[(('out_file', get_component_index_resampled), 'component_index')])])
    func_ntwk.connect([(matching_classification, group_graphs_anticorr,[('stats_file', 'matching_stats')])])
    func_ntwk.connect([(group_graphs_anticorr, grouped_graphs_anticorr,[('out_file', 'in_files')])])
    func_ntwk.connect([(grouped_graphs_anticorr, anticorrelationCFFConverter,[('out_files', 'gpickled_networks')])])
    func_ntwk.connect([(grouped_graphs_anticorr, remove_unconnected_anticorr,[('out_files', 'in_files')])])
    func_ntwk.connect([(grouped_graphs_anticorr, ConnectivityGraphNetworkXMetrics_anticorrelation,[('out_files', 'in_file')])])
    func_ntwk.connect([(grouped_graphs_anticorr, MergeCSVFiles_anticor,[(('out_files', pull_template_name), 'extra_field')])])
    func_ntwk.connect([(ConnectivityGraphNetworkXMetrics_anticorrelation, Matlab2CSV_nx_anticor,[("node_measures_matlab","in_file")])])
    func_ntwk.connect([(Matlab2CSV_nx_anticor, MergeCSVFiles_anticor,[("csv_files","in_files")])])
    func_ntwk.connect([(MergeCSVFiles_anticor, concatcsv_anticor,[("csv_file","in_files")])])
    func_ntwk.connect([(concatcsv_anticor, add_subjid_to_csv_anticor,[("out_name","in_file")])])
    func_ntwk.connect([(inputnode_within, add_subjid_to_csv_anticor,[("subject_id","extra_field")])])
    func_ntwk.connect([(inputnode_within, add_subjid_to_csv_anticor,[("subject_id","out_file")])])
    ConnectivityGraphNetworkXMetrics_correlation.inputs.ignore_exception = True
    ConnectivityGraphNetworkXMetrics_anticorrelation.inputs.ignore_exception = True

    # Create a higher-level workflow
    inputnode = pe.Node(interface=util.IdentityInterface(fields=["subject_id", "functional_images", "fmri_ICA_maps", "ica_mask_image", "fmri_ICA_timecourse", "segmentation_file", "repetition_time", "resolution_network_file"]), name="inputnode")


    if with_simple_timecourse_correlation:
        outputnode = pe.Node(interface = util.IdentityInterface(fields=["matching_stats", "neuronal_ntwks", "neuronal_cff", "neuronal_regional_timecourse_stats", "correlation_ntwks", "correlation_cff",
        "anticorrelation_ntwks", "anticorrelation_cff", "correlation_stats", "anticorrelation_stats", "simple_correlation_ntwks", "simple_correlation_cff"]), name="outputnode")
    else:
        outputnode = pe.Node(interface = util.IdentityInterface(fields=["matching_stats", "neuronal_ntwks", "neuronal_cff", "neuronal_regional_timecourse_stats", "correlation_ntwks", "correlation_cff",
        "correlation_stats", "anticorrelation_stats", "anticorrelation_ntwks", "anticorrelation_cff"]), name="outputnode")

    functional = pe.Workflow(name=name)
    functional.base_output_dir=name
    functional.base_dir=name
    functional.connect([(inputnode, func_ntwk, [("subject_id", "inputnode_within.subject_id"),
                                              ("functional_images", "inputnode_within.functional_images"),
                                              ("segmentation_file", "inputnode_within.segmentation_file"),
                                              ("repetition_time", "inputnode_within.repetition_time"),
                                              ("resolution_network_file", "inputnode_within.resolution_network_file")])
                                              ])

    functional.connect([(func_ntwk, outputnode,[('grouped_graphs.out_files', 'neuronal_ntwks')])])
    functional.connect([(func_ntwk, outputnode,[('neuronalCFFConverter.connectome_file', 'neuronal_cff')])])
    functional.connect([(func_ntwk, outputnode,[('neuronal_regional_timecourses.stats_file', 'neuronal_regional_timecourse_stats')])])

    functional.connect([(func_ntwk, outputnode,[('remove_unconnected_corr.out_files', 'correlation_ntwks')])])
    functional.connect([(func_ntwk, outputnode,[('correlationCFFConverter.connectome_file', 'correlation_cff')])])
    functional.connect([(func_ntwk, outputnode,[('add_subjid_to_csv_cor.csv_file', 'correlation_stats')])])

    functional.connect([(func_ntwk, outputnode,[('remove_unconnected_anticorr.out_files', 'anticorrelation_ntwks')])])
    functional.connect([(func_ntwk, outputnode,[('anticorrelationCFFConverter.connectome_file', 'anticorrelation_cff')])])
    functional.connect([(func_ntwk, outputnode,[('add_subjid_to_csv_anticor.csv_file', 'anticorrelation_stats')])])

    functional.connect([(func_ntwk, outputnode,[('matching_classification.stats_file', 'matching_stats')])])

    if with_simple_timecourse_correlation:
        functional.connect([(func_ntwk, outputnode,[('mergeSTCC.out', 'simple_correlation_ntwks')])])
        functional.connect([(func_ntwk, outputnode,[('TCcorrCFFConverter.connectome_file', 'simple_correlation_cff')])])
    return functional

def create_fmri_graph_grouping_workflow(data_dir, output_dir, name='averaging'):
    l2pipeline = pe.Workflow(name=name)
    l2pipeline.base_output_dir = output_dir
    type_list = ['correlation', 'anticorrelation']
    graph_list = ['Auditory', 'Cerebellum', 'DMN', 'ECN_L', 'ECN_R',  'Salience', 'Sensorimotor', 'Visual_lateral', 'Visual_medial', 'Visual_occipital']
    l2infosource = pe.Node(interface=util.IdentityInterface(fields=['type', 'graph']),
                         name="l2infosource")
    l2infosource.iterables = [('type', type_list), ('graph', graph_list)]
    l2source = pe.Node(nio.DataGrabber(infields=['type', 'graph'], outfields=['graph', 'networks', 'resolution_network_file']), name='l2source')

    template_arg_dict = {}
    template_arg_dict['networks'] = [['type', 'graph']]

    field_template_dict = {}
    field_template_dict['networks'] = op.join(output_dir,'%s/*/*%s*.pck')

    l2source.inputs.template_args = template_arg_dict
    l2source.inputs.base_directory = data_dir
    l2source.inputs.template = '%s/%s'
    l2source.inputs.field_template = field_template_dict
    l2source.inputs.raise_on_empty = False

    l2inputnode = pe.Node(interface=util.IdentityInterface(fields=['graph', 'networks', 'resolution_network_file']), name='l2inputnode')

    # Define a simple interface for a function which removes graphs with zero edges
    remove_unconnected_avg_and_cff_interface = Function(input_names=["in_files", "resolution_network_file", "group_id"],
                             output_names=["out_files"],
                             function=remove_unconnected_graphs_avg_and_cff)

    remove_unconnected_avg_and_cff = pe.Node(interface=remove_unconnected_avg_and_cff_interface, name='remove_unconnected_avg_and_cff')

    remove_unconnected_graphs_and_threshold_interface = Function(input_names=["in_file"],
                             output_names=["filtered_network_file"],
                             function=remove_unconnected_graphs_and_threshold)

    remove_unconnected_and_threshold = pe.MapNode(interface=remove_unconnected_graphs_and_threshold_interface,
                            name='remove_unconnected_and_threshold', iterfield=['in_file'])

    nxstats_and_merge_interface = Function(input_names=["network_file", "extra_field"],
                             output_names=["stats_file"],
                             function=nxstats_and_merge_csvs)

    nxstats_and_merge = pe.MapNode(interface=nxstats_and_merge_interface, name='nxstats_and_merge', iterfield=['network_file'])

    l2datasink = pe.Node(interface=nio.DataSink(), name="l2datasink")
    l2datasink.inputs.base_directory = output_dir

    l2pipeline = pe.Workflow(name="l2output")
    l2pipeline.base_dir = op.join(output_dir, 'l2output')

    l2pipeline.connect([
                        (l2infosource,l2source,[('type', 'type')]),
                        (l2source,l2inputnode,[('networks','networks')]),
                    ])

    l2pipeline.connect([
                        (l2infosource,l2source,[('graph','graph')]),
                        (l2infosource,l2inputnode,[('graph','graph')]),
                        #(l2inputnode,remove_unconnected_and_threshold,[('networks','in_file')]),
                        #(remove_unconnected_and_threshold,nxstats_and_merge,[('filtered_network_file','network_file')]),
                        #(l2inputnode,nxstats_and_merge,[('graph','extra_field')]),
                        #(remove_unconnected_and_threshold,remove_unconnected_avg_and_cff,[('filtered_network_file','in_files')]),
                        (l2inputnode,remove_unconnected_avg_and_cff,[('networks','in_files')]),
                        (l2inputnode,remove_unconnected_avg_and_cff,[('resolution_network_file','resolution_network_file')]),
                        (l2inputnode,remove_unconnected_avg_and_cff,[('graph','group_id')]),
                    ])

    return l2pipeline
