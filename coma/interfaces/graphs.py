from nipype.interfaces.base import (BaseInterface, traits,
                                    File, TraitedSpec, InputMultiPath,
                                    isdefined)
from nipype.utils.filemanip import split_filename
import os.path as op
import numpy as np
import nibabel as nb
import networkx as nx
import scipy.io as sio
from nipype.workflows.misc.utils import get_data_dims
from nipype.interfaces.cmtk.nx import (remove_all_edges, add_node_data, add_edge_data)
from nipype import logging
iflogger = logging.getLogger('interface')


class CreateConnectivityThresholdInputSpec(TraitedSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True, xor=[
                              'in_file4d'], desc='Original functional magnetic resonance image (fMRI) as a set of 3-dimensional images')
    in_file4d = File(exists=True, mandatory=True, xor=[
                     'in_files'], desc='Original functional magnetic resonance image (fMRI) as a 4-dimension image')
    time_course_file = File(exists=True, mandatory=True,
                            desc='Independent component timecourse array (as an image)')
    segmentation_file = File(exists=True, mandatory=True,
                             desc='Image with segmented regions (e.g. aparc+aseg.nii or the output from cmtk.Parcellate())')
    subject_id = traits.Str(desc='Subject ID')
    out_t_value_threshold_file = File(
        'tvalues.mat', usedefault=True, desc='T values per node saved as a Matlab .mat')


class CreateConnectivityThresholdOutputSpec(TraitedSpec):
    t_value_threshold_file = File(
        desc='Output matlab file containing the thresholding parameters for each region/node')


class CreateConnectivityThreshold(BaseInterface):
    input_spec = CreateConnectivityThresholdInputSpec
    output_spec = CreateConnectivityThresholdOutputSpec

    def _run_interface(self, runtime):
        iflogger.info(
            'Segmentation image: {img}'.format(img=self.inputs.segmentation_file))
        rois = get_roi_list(self.inputs.segmentation_file)
        number_of_nodes = len(rois)
        iflogger.info('Found {roi} unique region values'.format(roi=len(rois)))

        if len(self.inputs.in_files) > 1:
            iflogger.info('Multiple input images detected')
            iflogger.info(len(self.inputs.in_files))
            in_files = self.inputs.in_files
            try:
                # VERY tempermental solution to sorting the functional images.
                # RELIES ON xyz-01_out.nii from recent resampling.
                in_files = sorted(in_files, key=lambda x:
                                  int(x.split("_")[-2].split("-")[-2]))
            except IndexError:
                # Tempermental solution to sorting the functional images.
                # RELIES ON xyz_out.nii from recent resampling.
                in_files = sorted(in_files, key=lambda x:
                                  int(x.split("_")[-2]))
        elif isdefined(self.inputs.in_file4d):
            iflogger.info('Single four-dimensional image selected')
            in_file4d = nb.load(self.inputs.in_file4d)
            in_files = nb.four_to_three(in_file4d)
        else:
            iflogger.info('Single functional image provided')
            in_files = self.inputs.in_files

        rois = get_roi_list(self.inputs.segmentation_file)
        fMRI_timecourse, _, _, _, _ = get_timecourse_by_region(
            in_files, self.inputs.segmentation_file, rois)
        timecourse_at_each_node = fMRI_timecourse.T
        time_course_image = nb.load(self.inputs.time_course_file)

        iflogger.info(np.shape(timecourse_at_each_node))

        number_of_components = 30
        number_of_images = 198
        time_course_data = time_course_image.get_data()
        onerow = np.ones(number_of_images)
        time_course_per_IC = np.vstack((onerow.T, time_course_data.T)).T
        iflogger.info(np.shape(time_course_per_IC))

        y = timecourse_at_each_node
        X = time_course_per_IC
        n = number_of_images
        XTX_inv = np.linalg.inv(np.dot(X.T, X))
        N = number_of_components + 1
        contrast = np.concatenate(
            (np.zeros((1, number_of_components)), np.eye(number_of_components))).T
        a = np.empty((number_of_nodes, N))
        residues = np.empty((number_of_nodes, 1))
        rank = np.empty((number_of_nodes, 1))
        singularvalues = np.empty((number_of_nodes, N))
        resids = np.empty((number_of_nodes, number_of_images))
        error_variance = np.empty((number_of_nodes, 1))
        t_values = np.empty((number_of_nodes, number_of_components))
        beta_value = np.empty((number_of_components, 1))
        for node_idx, node in enumerate(rois):
            a[node_idx], residues[node_idx], rank[node_idx], singularvalues[
                node_idx] = np.linalg.lstsq(X, y[:, node_idx])
            resids[node_idx, :] = y[:, node_idx] - \
                np.dot(X, a[node_idx])       # e = y - Xa;
            error_variance[node_idx] = np.var(resids[node_idx])
            for IC in range(0, number_of_components):
                t_values[node_idx, IC] = np.dot(contrast[IC], a[node_idx]) / \
                    np.sqrt(
                        error_variance[node_idx] * np.dot(np.dot(contrast[IC], XTX_inv), contrast[IC, :].T))
                beta_value[IC] = np.dot(contrast[IC], a[node_idx])

        t_value_dict = {}
        t_value_dict['t_value_per_node'] = t_values
        t_value_dict['timecourse_at_each_node'] = y
        t_value_dict['timecourse_per_IC'] = X
        t_value_dict['number_of_images'] = n
        t_value_dict['a'] = a
        t_value_dict['contrast'] = contrast
        t_value_dict['residuals'] = resids
        t_value_dict['error_variance'] = error_variance

        out_file = op.abspath(self.inputs.out_t_value_threshold_file)
        iflogger.info(
            'Saving T-values per node, per IC, as {file}'.format(file=out_file))
        sio.savemat(out_file, t_value_dict)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        out_file = op.abspath(self.inputs.out_t_value_threshold_file)
        outputs['t_value_threshold_file'] = out_file
        return outputs


class ConnectivityGraphInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc='fMRI ICA Map')
    resolution_network_file = File(
        exists=True, mandatory=True, desc='Network resolution file')
    t_value_threshold_file = File(
        exists=True, mandatory=True, desc='T-value threshold per node per IC. Saved as a Matlab .mat file.')
    component_index = traits.Int(
        mandatory=True, desc='Index of the independent component to use from the t-value threshold file.')
    segmentation_file = File(
        exists=True, desc='Image with segmented regions (e.g. aparc+aseg.nii or the output from cmtk.Parcellate())')
    subject_id = traits.Str(desc='Subject ID')
    give_nodes_values = traits.Bool(
        False, desc='Controls whether or not nodes are given scalar values from the functional image')
    significance_threshold = traits.Float(
        0.001, usedefault=True, desc='Significance threshold used to keep an edge.')
    number_of_images = traits.Int(
        198, usedefault=True, desc='Number of functional images used to generate the threshold file')
    out_stats_file = File(
        desc='Some simple image statistics for regions saved as a Matlab .mat')
    out_network_file = File(desc='The output network as a NetworkX gpickle.')


class ConnectivityGraphOutputSpec(TraitedSpec):
    stats_file = File(
        desc='Some simple image statistics for the original and normalized images saved as a Matlab .mat')
    network_file = File(
        desc='Output gpickled network file for the connectivity graph.')
    correlation_network = File(
        desc='Output gpickled network file for the correlation network.')
    anticorrelation_network = File(
        desc='Output gpickled network file for the anticorrelation network.')


class ConnectivityGraph(BaseInterface):

    """
    Creates a weighted connectivity graph given a 4D functional image or list of functional images given a segmentated image.
    Output is saved in a MATLAB file, and if a network resolution file is provided (e.g. resolution1015.graphml), the regions are output as nodes in a NetworkX graph.

    Example
    -------

    >>> import nipype.interfaces.cmtk as cmtk
    >>> congraph = cmtk.ConnectivityGraph()
    >>> congraph.inputs.in_file = 'pet_resliced_1.nii'
    >>> congraph.inputs.segmentation_file = 'ROI_scale500.nii.gz'
    >>> congraph.run() # doctest: +SKIP
    """
    input_spec = ConnectivityGraphInputSpec
    output_spec = ConnectivityGraphOutputSpec

    def _run_interface(self, runtime):
        key = 'congraph'
        edge_key = 'weight'

        iflogger.info(
            'T-value Threshold file: {t}'.format(t=self.inputs.t_value_threshold_file))
        iflogger.info(
            'Independent component to use: {i}'.format(i=self.inputs.component_index))
        path, name, ext = split_filename(self.inputs.t_value_threshold_file)

        if ext == '.mat':
            t_value_dict = sio.loadmat(self.inputs.t_value_threshold_file)
            t_values = t_value_dict['t_value_per_node']
            t_value_per_node = t_values[:, self.inputs.component_index - 1]
            number_of_ICs = np.shape(t_values)[1]
        else:
            iflogger.info(
                "Please save the t-values as a Matlab file with key 't_value_per_node'")

        functional = nb.load(self.inputs.in_file)
        functionaldata = functional.get_data()
        segmentation = nb.load(self.inputs.segmentation_file)
        segmentationdata = segmentation.get_data()
        rois = get_roi_list(self.inputs.segmentation_file)
        number_of_nodes = len(rois)
        iflogger.info(
            'Found {roi} unique region values'.format(roi=number_of_nodes))
        iflogger.info('Significance threshold: {p}'.format(
            p=self.inputs.significance_threshold))
        #number_of_images = self.inputs.number_of_images
        # degrees_of_freedom = number_of_images - \
        #    number_of_ICs - 1
        #sig = self.inputs.significance_threshold
        #threshold = tinv((1-sig), degrees_of_freedom)
        #threshold = 2*threshold
        #iflogger.info('Weight threshold: {w}'.format(w=threshold))

        iflogger.info(
            'Functional image: {img}'.format(img=self.inputs.in_file))
        iflogger.info(
            'Segmentation image: {img}'.format(img=self.inputs.segmentation_file))
        if not get_data_dims(self.inputs.in_file) == get_data_dims(self.inputs.segmentation_file):
            iflogger.error(
                'Image dimensions are not the same, please reslice the images to the same dimensions')
            dx, dy, dz = get_data_dims(self.inputs.in_file)
            iflogger.error('Functional image dimensions: {dimx}, {dimy}, {dimz}'.format(
                dimx=dx, dimy=dy, dimz=dz))
            dx, dy, dz = get_data_dims(self.inputs.segmentation_file)
            iflogger.error('Segmentation image dimensions: {dimx}, {dimy}, {dimz}'.format(
                dimx=dx, dimy=dy, dimz=dz))

        stats = {}

        if self.inputs.give_nodes_values:
            func_mean = []
            for idx, roi in enumerate(rois):
                values = []
                x, y, z = np.where(segmentationdata == roi)
                for index in range(0, len(x)):
                    value = functionaldata[x[index]][y[index]][z[index]]
                    values.append(value)
                func_mean.append(np.mean(values))
                iflogger.info(
                    'Region ID: {id}, Mean Value: {avg}'.format(id=roi, avg=np.mean(values)))
            stats[key] = func_mean

        connectivity_matrix = np.zeros((number_of_nodes, number_of_nodes))
        correlation_matrix = np.zeros((number_of_nodes, number_of_nodes))
        anticorrelation_matrix = np.zeros((number_of_nodes, number_of_nodes))
        iflogger.info('Drawing edges...')
        for idx_i, roi_i in enumerate(rois):
            t_i = t_value_per_node[idx_i]
            #iflogger.info('ROI:{i}, T-value: {t}'.format(i=roi_i, t=t_i))
            for idx_j, roi_j in enumerate(rois):
                t_j = t_value_per_node[idx_j]
                #iflogger.info('...ROI:{j}, T-value: {t}'.format(j=roi_j, t=t_j))
                if idx_j > idx_i:
                    if (t_i > 0 and t_j > 0) or (t_i < 0 and t_j < 0):
                        weight = abs(t_i) + abs(t_j) - abs(t_i - t_j)
                        #iflogger.info('Weight = {w}     T-values for ROIs {i}-{j}'.format(w=weight, i=t_i, j=t_j))
                        # if weight > threshold:
                        connectivity_matrix[idx_i, idx_j] = weight
                        correlation_matrix[idx_i, idx_j] = weight
                            #iflogger.info('Drawing a correlation edge for ROIs {i}-{j} at {id_i},{id_j}'.format(i=roi_i, j=roi_j, id_i=idx_i, id_j=idx_j))
                    elif (t_i < 0 and t_j > 0) or (t_i > 0 and t_j < 0):
                        weight = abs(t_i) + abs(t_j) - abs(t_i + t_j)
                        #iflogger.info('Weight = {w}     T-values for ROIs {i}-{j}'.format(w=weight, i=t_i, j=t_j))
                        # if weight > threshold:
                        connectivity_matrix[idx_i, idx_j] = -weight
                        anticorrelation_matrix[idx_i, idx_j] = weight
                            #iflogger.info('Drawing an anticorrelation edge for ROIs {i}-{j} at {id_i},{id_j}'.format(i=roi_i, j=roi_j, id_i=idx_i, id_j=idx_j))
                            #iflogger.info('Weight = {w}     T-values for ROIs {i}-{j}'.format(w=weight, i=t_i, j=t_j))

        edges = len(np.nonzero(connectivity_matrix)[0])
        cor_edges = len(np.nonzero(correlation_matrix)[0])
        anticor_edges = len(np.nonzero(anticorrelation_matrix)[0])

        iflogger.info('Total edges: {e}'.format(e=edges))
        iflogger.info('Total correlation edges: {c}'.format(c=cor_edges))
        iflogger.info(
            'Total anticorrelation edges: {a}'.format(a=anticor_edges))

        connectivity_matrix = connectivity_matrix + connectivity_matrix.T
        correlation_matrix = correlation_matrix + correlation_matrix.T
        anticorrelation_matrix = anticorrelation_matrix + \
            anticorrelation_matrix.T

        stats[edge_key] = connectivity_matrix
        stats['correlation'] = correlation_matrix
        stats['anticorrelation'] = anticorrelation_matrix

        try:
            gp = nx.read_gpickle(self.inputs.resolution_network_file)
        except IndexError:
            gp = nx.read_graphml(self.inputs.resolution_network_file)
        nodedict = gp.node[gp.nodes()[0]]
        if not nodedict.has_key('dn_position'):
            iflogger.info("Creating node positions from segmentation")
            G = nx.Graph()
            for u, d in gp.nodes_iter(data=True):
                G.add_node(int(u), d)
                xyz = tuple(
                    np.mean(np.where(np.flipud(segmentationdata) == int(d["dn_correspondence_id"])), axis=1))
                G.node[int(u)]['dn_position'] = xyz
            ntwkname = op.abspath('nodepositions.pck')
            nx.write_gpickle(G, ntwkname)
        else:
            ntwkname = self.inputs.resolution_network_file

        try:
            ntwkname = nx.read_gpickle(ntwkname)
        except IndexError:
            ntwkname = nx.read_graphml(ntwkname)

        newntwk = ntwkname.copy()
        newntwk = remove_all_edges(newntwk)

        if self.inputs.give_nodes_values:
            newntwk = add_node_data(stats[key], newntwk)
            corntwk = add_node_data(stats[key], newntwk)
            anticorntwk = add_node_data(stats[key], newntwk)

            newntwk = add_edge_data(stats[edge_key], newntwk)
            corntwk = add_edge_data(stats['correlation'], corntwk)
            anticorntwk = add_edge_data(stats['anticorrelation'], anticorntwk)
        else:
            newntwk = add_edge_data(stats[edge_key], ntwkname)
            corntwk = add_edge_data(stats['correlation'], ntwkname)
            anticorntwk = add_edge_data(stats['anticorrelation'], ntwkname)

        if isdefined(self.inputs.out_network_file):
            path, name, ext = split_filename(self.inputs.out_network_file)
            if not ext == '.pck':
                ext = '.pck'
            out_network_file = op.abspath(name + ext)
        else:
            if isdefined(self.inputs.subject_id):
                out_network_file = op.abspath(
                    self.inputs.subject_id + '_IC_' + str(self.inputs.component_index) + '.pck')
            else:
                out_network_file = op.abspath(
                    'IC_' + str(self.inputs.component_index) + '.pck')

        path, name, ext = split_filename(out_network_file)
        iflogger.info(
            'Saving output network as {ntwk}'.format(ntwk=out_network_file))
        nx.write_gpickle(newntwk, out_network_file)

        out_correlation_network = op.abspath(name + '_correlation' + ext)
        iflogger.info(
            'Saving correlation network as {ntwk}'.format(ntwk=out_correlation_network))
        nx.write_gpickle(corntwk, out_correlation_network)

        out_anticorrelation_network = op.abspath(
            name + '_anticorrelation' + ext)
        iflogger.info('Saving anticorrelation network as {ntwk}'.format(
            ntwk=out_anticorrelation_network))
        nx.write_gpickle(anticorntwk, out_anticorrelation_network)

        if isdefined(self.inputs.subject_id):
            stats['subject_id'] = self.inputs.subject_id

        if isdefined(self.inputs.out_stats_file):
            path, name, ext = split_filename(self.inputs.out_stats_file)
            if not ext == '.mat':
                ext = '.mat'
            out_stats_file = op.abspath(name + ext)
        else:
            if isdefined(self.inputs.subject_id):
                out_stats_file = op.abspath(
                    self.inputs.subject_id + '_IC_' + str(self.inputs.component_index) + '.mat')
            else:
                out_stats_file = op.abspath(
                    'IC_' + str(self.inputs.component_index) + '.mat')

        iflogger.info(
            'Saving image statistics as {stats}'.format(stats=out_stats_file))
        sio.savemat(out_stats_file, stats)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.out_stats_file):
            path, name, ext = split_filename(self.inputs.out_stats_file)
            if not ext == '.pck':
                ext = '.pck'
            out_stats_file = op.abspath(name + ext)
        else:
            if isdefined(self.inputs.subject_id):
                out_stats_file = op.abspath(
                    self.inputs.subject_id + '_IC_' + str(self.inputs.component_index) + '.mat')
            else:
                out_stats_file = op.abspath(
                    'IC_' + str(self.inputs.component_index) + '.mat')
        outputs["stats_file"] = out_stats_file

        if isdefined(self.inputs.out_network_file):
            path, name, ext = split_filename(self.inputs.out_network_file)
            if not ext == '.pck':
                ext = '.pck'
            out_network_file = op.abspath(name + ext)
        else:
            if isdefined(self.inputs.subject_id):
                out_network_file = op.abspath(
                    self.inputs.subject_id + '_IC_' + str(self.inputs.component_index) + '.pck')
            else:
                out_network_file = op.abspath(
                    'IC_' + str(self.inputs.component_index) + '.pck')
        outputs["network_file"] = out_network_file
        path, name, ext = split_filename(out_network_file)
        out_correlation_network = op.abspath(name + '_correlation' + ext)
        outputs["correlation_network"] = out_correlation_network
        out_anticorrelation_network = op.abspath(
            name + '_anticorrelation' + ext)
        outputs["anticorrelation_network"] = out_anticorrelation_network
        return outputs
