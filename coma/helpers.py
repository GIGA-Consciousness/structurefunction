def pull_template_name(in_files):
    from nipype.utils.filemanip import split_filename
    out_files = []
    graph_list = ['Auditory', 'Cerebellum', 'DMN', 'ECN_L', 'ECN_R',  'Salience', 'Sensorimotor', 'Visual_lateral', 'Visual_medial', 'Visual_occipital']
    for in_file in in_files:
        found = False
        path, name, ext = split_filename(in_file)
        for graph_name in graph_list:
            if name.rfind(graph_name) > 0:
                out_files.append(graph_name)
                found = True
        if not found:
            out_files.append(name)
    assert len(in_files) == len(out_files)
    return out_files



def get_component_index_resampled(out):
    if isinstance(out, str):
        idx = int(out.split('_')[-2])
    elif isinstance(out, list):
        idx = []
        for value in out:
            idx_value = int(value.split('_')[-2])
            idx.append(idx_value)
    return idx


def get_component_index(out):
    if isinstance(out, str):
        idx = int(out.split('_')[-1].split('.')[0])
    elif isinstance(out, list):
        idx = []
        for value in out:
            idx_value = int(value.split('_')[-1].split('.')[0])
            idx.append(idx_value)
    return idx


def nxstats_and_merge_csvs(network_file, extra_field):
    from nipype.workflows.dmri.connectivity.nx import create_networkx_pipeline
    if network_file != [None]:
        nxstats = create_networkx_pipeline(name="networkx", extra_column_heading="Template")
        nxstats.inputs.inputnode.network_file = network_file
        nxstats.inputs.inputnode.extra_field = extra_field
        nxstats.run()
    stats_file = ""
    return stats_file


def remove_unconnected_graphs(in_files):
    import networkx as nx
    out_files = []
    if in_files == None:
        return None
    elif len(in_files) == 0:
        return None
    for in_file in in_files:
        graph = nx.read_gpickle(in_file)
        if not graph.number_of_edges() == 0:
            out_files.append(in_file)
    return out_files

def remove_unconnected_graphs_and_threshold(in_file):
    import nipype.interfaces.cmtk as cmtk
    import nipype.pipeline.engine as pe
    import os, os.path as op
    import networkx as nx
    from nipype.utils.filemanip import split_filename
    connected = []
    if in_file == None or in_file == [None]:
        return None
    elif len(in_file) == 0:
        return None
    graph = nx.read_gpickle(in_file)
    if not graph.number_of_edges() == 0:
        connected.append(in_file)
        _, name, ext = split_filename(in_file)
        filtered_network_file = op.abspath(name + '_filt' + ext)
    if connected == []:
        return None

    #threshold_graphs = pe.Node(interface=cmtk.ThresholdGraph(), name="threshold_graphs")
    threshold_graphs = cmtk.ThresholdGraph()
    from nipype.interfaces.cmtk.functional import tinv
    weight_threshold = 1 #tinv(0.95, 198-30-1)
    threshold_graphs.inputs.network_file = in_file
    threshold_graphs.inputs.weight_threshold = weight_threshold
    threshold_graphs.inputs.above_threshold = True
    threshold_graphs.inputs.edge_key = "weight"
    threshold_graphs.inputs.out_filtered_network_file = op.abspath(filtered_network_file)
    threshold_graphs.run()
    return op.abspath(filtered_network_file)

def remove_unconnected_graphs_avg_and_cff(in_files, resolution_network_file, group_id):
    import nipype.interfaces.cmtk as cmtk
    import nipype.pipeline.engine as pe
    from nipype.utils.filemanip import split_filename
    import os, os.path as op
    import networkx as nx
    connected = []
    if in_files == None or in_files == [None]:
        return None
    elif len(in_files) == 0:
        return None
    for in_file in in_files:
        graph = nx.read_gpickle(in_file)
        if not graph.number_of_edges() == 0:
            connected.append(in_file)
            print in_file
    if connected == []:
        return None

    #import ipdb
    #ipdb.set_trace()
    avg_out_name = op.abspath(group_id + '_n=' + str(len(connected)) + '_average.pck')
    avg_out_cff_name = op.abspath(group_id + '_n=' + str(len(connected)) + '_Networks.cff')

    average_networks = cmtk.AverageNetworks()
    average_networks.inputs.in_files = connected
    average_networks.inputs.resolution_network_file = resolution_network_file
    average_networks.inputs.group_id = group_id
    average_networks.inputs.out_gpickled_groupavg = avg_out_name
    average_networks.run()

    _, name, ext = split_filename(avg_out_name)
    filtered_network_file = op.abspath(name + '_filt' + ext)

    threshold_graphs = cmtk.ThresholdGraph()
    from nipype.interfaces.cmtk.functional import tinv
    weight_threshold = 1 #tinv(0.95, 198-30-1)
    threshold_graphs.inputs.network_file = avg_out_name
    threshold_graphs.inputs.weight_threshold = weight_threshold
    threshold_graphs.inputs.above_threshold = True
    threshold_graphs.inputs.edge_key = "value"
    threshold_graphs.inputs.out_filtered_network_file = op.abspath(filtered_network_file)
    threshold_graphs.run()

    out_files = []
    out_files.append(avg_out_name)
    out_files.append(op.abspath(filtered_network_file))
    out_files.extend(connected)

    average_cff = cmtk.CFFConverter()
    average_cff.inputs.gpickled_networks = out_files
    average_cff.inputs.out_file = avg_out_cff_name
    average_cff.run()

    out_files.append(op.abspath(avg_out_cff_name))

    return out_files