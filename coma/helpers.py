import os.path as op
import gzip
import nibabel as nb
import numpy as np
from nipype.utils.filemanip import split_filename

def nifti_to_analyze(nii):
    nifti = nb.load(nii)
    if nii[-3:] == '.gz':
        nif = gzip.open(nii, 'rb')
    else:
        nif = open(nii, 'rb')
    hdr = nb.nifti1.Nifti1Header.from_fileobj(nif)

    arr_hdr = nb.analyze.AnalyzeHeader.from_header(hdr)
    arrb = hdr.raw_data_from_fileobj(nif)
    img = nb.AnalyzeImage(
        dataobj=arrb, affine=nifti.get_affine(), header=arr_hdr)
    _, name, _ = split_filename(nii)
    nb.analyze.save(img, op.abspath(name + '.img'))
    return op.abspath(name + '.img'), op.abspath(name + '.hdr')


def analyze_to_nifti(img, ext='.nii.gz', affine=None):
    image = nb.load(img)
    _, name, _ = split_filename(img)
    if affine is None:
        nii = nb.Nifti1Image.from_image(image)
        affine = image.get_affine()
        nii.set_sform(affine)
        nii.set_qform(affine)
    else:
        nii = nb.Nifti1Image(dataobj=image.get_data(),
                             header=image.get_header(), affine=affine)

    nb.save(nii, op.abspath(name + ext))
    return op.abspath(name + ext)


def switch_datatype(in_file, dt=np.uint8):
    '''
    Changes ROI values to prevent values equal to 1, 2,
    or 3. These are reserved for GM/WM/CSF in the PVELab
    functions.
    '''

    image = nb.load(in_file)
    image.set_data_dtype(dt)
    _, name, _ = split_filename(in_file)
    fixed_image = op.abspath(name + "_u8.nii.gz")
    nb.save(image, fixed_image)
    return fixed_image


def get_names(lookup_table):
    LUT_dict = {}
    with open(lookup_table) as LUT:
        for line in LUT:
            if line[0] != "#" and line != "\r\n":
                parse = line.split()
                LUT_dict[int(parse[0])] = parse[1].replace(" ", "")
    return LUT_dict


def prepare_for_uint8(in_array, ignore=[0]):
    import numpy as np
    np.unique(in_array)
    assert(in_array.all > 0)
    assert(in_array.any > 255)
    uniquevals = np.unique(in_array)
    uniquevals = uniquevals.tolist()
    uniquevals.sort()

    for ig in ignore:
        if ig in uniquevals:
            uniquevals.remove(ig)
    assert(len(uniquevals) < 204)
    # We start at 51 because PVElab doesn't work otherwise. No idea why
    # We first remap to negative values so we don't accidentally
    # overwrite previously changed values
    out_data = in_array.copy()
    remap_dict = {}
    for idx, v in enumerate(xrange(51, len(uniquevals) + 51)):
        old = uniquevals[idx]
        remap_dict[v] = old
        out_data[np.where(in_array == old)] = -v
    out_data = out_data * -1
    return out_data, remap_dict


def pull_template_name(in_files):
    from nipype.utils.filemanip import split_filename
    out_files = []
    graph_list = [
        'Auditory', 'Cerebellum', 'DMN', 'ECN_L', 'ECN_R',  'Salience',
        'Sensorimotor', 'Visual_lateral', 'Visual_medial', 'Visual_occipital']
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
        nxstats = create_networkx_pipeline(
            name="networkx", extra_column_heading="Template")
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
    import os
    import os.path as op
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
    weight_threshold = 1  # tinv(0.95, 198-30-1)
    threshold_graphs.inputs.network_file = in_file
    threshold_graphs.inputs.weight_threshold = weight_threshold
    threshold_graphs.inputs.above_threshold = True
    threshold_graphs.inputs.edge_key = "weight"
    threshold_graphs.inputs.out_filtered_network_file = op.abspath(
        filtered_network_file)
    threshold_graphs.run()
    return op.abspath(filtered_network_file)


def remove_unconnected_graphs_avg_and_cff(in_files, resolution_network_file, group_id):
    import nipype.interfaces.cmtk as cmtk
    import nipype.pipeline.engine as pe
    from nipype.utils.filemanip import split_filename
    import os
    import os.path as op
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

    avg_out_name = op.abspath(
        group_id + '_n=' + str(len(connected)) + '_average.pck')
    avg_out_cff_name = op.abspath(
        group_id + '_n=' + str(len(connected)) + '_Networks.cff')

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
    weight_threshold = 1  # tinv(0.95, 198-30-1)
    threshold_graphs.inputs.network_file = avg_out_name
    threshold_graphs.inputs.weight_threshold = weight_threshold
    threshold_graphs.inputs.above_threshold = True
    threshold_graphs.inputs.edge_key = "value"
    threshold_graphs.inputs.out_filtered_network_file = op.abspath(
        filtered_network_file)
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


def add_subj_name_to_sfmask(subject_id):
    return subject_id + "_SingleFiberMask.nii.gz"


def add_subj_name_to_fdgpet(subject_id):
    return subject_id + "_fdgpet.nii"


def add_subj_name_to_wmmask(subject_id):
    return subject_id + "_wmmask.nii"


def add_subj_name_to_termmask(subject_id):
    return subject_id + "_cortex.nii"


def add_subj_name_to_T1brain(subject_id):
    return subject_id + "_T1brain.nii"


def add_subj_name_to_T1(subject_id):
    return subject_id + "_T1.nii"


def add_subj_name_to_rois(subject_id):
    return subject_id + "_PreCoTh_rois.nii"


def select_CSF(tissue_class_files):
    CSF = None
    for in_file in tissue_class_files:
        if in_file.rfind("_seg_0") > 0 or in_file.rfind("_pve_0") > 0:
            CSF = in_file
    return CSF


def select_GM(tissue_class_files):
    GM = None
    for in_file in tissue_class_files:
        if in_file.rfind("_seg_1") > 0 or in_file.rfind("_pve_1") > 0:
            GM = in_file
    return GM


def select_WM(tissue_class_files):
    WM = None
    for in_file in tissue_class_files:
        if in_file.rfind("_seg_2") > 0 or in_file.rfind("_pve_2") > 0:
            WM = in_file
    return WM


def return_subject_data(subject_id, data_file):
    import csv
    from nipype import logging
    iflogger = logging.getLogger('interface')

    f = open(data_file, 'r')
    csv_line_by_line = csv.reader(f)
    found = False
    # Must be stored in this order!:
    # 'subject_id', 'dose', 'weight', 'delay', 'glycemie', 'scan_time']
    for line in csv_line_by_line:
        if line[0] == subject_id:
            dose, weight, delay, glycemie, scan_time = [
                float(x) for x in line[1:]]
            iflogger.info('Subject %s found' % subject_id)
            iflogger.info('Dose: %s' % dose)
            iflogger.info('Weight: %s' % weight)
            iflogger.info('Delay: %s' % delay)
            iflogger.info('Glycemie: %s' % glycemie)
            iflogger.info('Scan Time: %s' % scan_time)
            found = True
            break
    if not found:
        raise Exception("Subject id %s was not in the data file!" % subject_id)
    return dose, weight, delay, glycemie, scan_time


def select_ribbon(list_of_files):
    from nipype.utils.filemanip import split_filename
    for in_file in list_of_files:
        _, name, ext = split_filename(in_file)
        if name == 'ribbon':
            idx = list_of_files.index(in_file)
    return list_of_files[idx]


def wm_labels_only(in_file, out_filename=None, include_thalamus=False):
    from nipype.utils.filemanip import split_filename
    import nibabel as nb
    import numpy as np
    import os.path as op
    in_image = nb.load(in_file)
    in_header = in_image.get_header()
    in_data = in_image.get_data()

    # Left and right cerebral WM
    out_data = np.zeros(np.shape(in_data))
    out_data[np.where(in_data == 2)] = 1
    out_data[np.where(in_data == 41)] = 1

    # Left and right WM hypointensities
    out_data[np.where(in_data == 81)] = 1
    out_data[np.where(in_data == 82)] = 1

    # Left and right cerebellar WM
    out_data[np.where(in_data == 7)] = 1
    out_data[np.where(in_data == 46)] = 1

    # Corpus callosum posterior to anterior
    out_data[np.where(in_data == 251)] = 1
    out_data[np.where(in_data == 252)] = 1
    out_data[np.where(in_data == 253)] = 1
    out_data[np.where(in_data == 254)] = 1
    out_data[np.where(in_data == 255)] = 1

    if include_thalamus:
        # Left and right thalami
        out_data[np.where(in_data == 49)] = 1
        out_data[np.where(in_data == 10)] = 1

    if out_filename is None:
        _, name, _ = split_filename(in_file)
        out_filename = name + "_wm.nii.gz"
    out_file = op.abspath(out_filename)
    try:
        out_image = nb.Nifti1Image(
            data=out_data, header=in_header, affine=in_image.get_affine())
    except TypeError:
        out_image = nb.Nifti1Image(
            dataobj=out_data, header=in_header, affine=in_image.get_affine())
    nb.save(out_image, out_file)
    return out_file


def csf_labels_only(in_file, out_filename=None):
    from nipype.utils.filemanip import split_filename
    import nibabel as nb
    import numpy as np
    import os.path as op
    in_image = nb.load(in_file)
    in_header = in_image.get_header()
    in_data = in_image.get_data()

    out_data = np.zeros(np.shape(in_data))

    # Left and right lateral ventricles
    out_data[np.where(in_data == 4)] = 1
    out_data[np.where(in_data == 43)] = 1

    # Left and right inferior lateral ventricles
    out_data[np.where(in_data == 5)] = 1
    out_data[np.where(in_data == 44)] = 1

    # Left and right choroid plexus
    out_data[np.where(in_data == 31)] = 1
    out_data[np.where(in_data == 63)] = 1

    # 3rd and 4th ventricles
    out_data[np.where(in_data == 14)] = 1
    out_data[np.where(in_data == 15)] = 1

    # 5th ventricle
    out_data[np.where(in_data == 72)] = 1

    # CSF
    out_data[np.where(in_data == 24)] = 1

    if out_filename is None:
        _, name, _ = split_filename(in_file)
        out_filename = name + "_csf.nii.gz"
    out_file = op.abspath(out_filename)
    try:
        out_image = nb.Nifti1Image(
            data=out_data, header=in_header, affine=in_image.get_affine())
    except TypeError:
        out_image = nb.Nifti1Image(
            dataobj=out_data, header=in_header, affine=in_image.get_affine())
    nb.save(out_image, out_file)
    return out_file
