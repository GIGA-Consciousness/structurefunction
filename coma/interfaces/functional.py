from nipype.interfaces.base import (BaseInterface, traits,
                                    File, TraitedSpec, InputMultiPath,
                                    OutputMultiPath, isdefined)
from nipype.utils.filemanip import split_filename
import os.path as op
import numpy as np
import nibabel as nb
import networkx as nx
import scipy.io as sio
from nipype.interfaces.cmtk.nx import (remove_all_edges, add_node_data, add_edge_data)
from scipy.stats.stats import pearsonr
from ..helpers import get_names

from nipype import logging
iflogger = logging.getLogger('interface')


def get_roi_list(segmentation_file):
    segmentation = nb.load(segmentation_file)
    segmentationdata = segmentation.get_data()
    rois = np.unique(segmentationdata)
    rois = list(rois)
    rois.sort()
    rois.remove(0)
    return rois


class RegionalValuesInputSpec(TraitedSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True, xor=[
                              'in_file4d'], desc='Functional (e.g. Positron Emission Tomography) image')
    in_file4d = File(exists=True, mandatory=True, xor=[
                     'in_files'], desc='Functional (e.g. Positron Emission Tomography) image')
    segmentation_file = File(exists=True, mandatory=True,
                             desc='Image with segmented regions (e.g. aparc+aseg.nii or the output from cmtk.Parcellate())')
    lookup_table = File(exists=True, desc='Optional lookup table for grabbing region names')
    resolution_network_file = File(exists=True, desc='Parcellation files from Connectome Mapping Toolkit. This is not necessary'
                                   ', but if included, the interface will output the statistical maps as networkx graphs.')
    subject_id = traits.Str(desc='Subject ID')
    skip_unknown = traits.Bool(
        True, usedefault=True, desc='Skips calculation for regions with ID = 0 (default=True)')
    out_stats_file = File('stats.mat', usedefault=True,
                          desc='Some simple image statistics for regions saved as a Matlab .mat')


class RegionalValuesOutputSpec(TraitedSpec):
    stats_file = File(
        desc='Some simple image statistics for the original and normalized images saved as a Matlab .mat')
    networks = OutputMultiPath(
        File(desc='Output gpickled network files for all statistical measures'))


class RegionalValues(BaseInterface):

    """
    Extracts the regional mean, max, min, and standard deviation for a single functional image, 4D functional image, or list of functional images given a segmentated image.
    Output is saved in a MATLAB file, and if a network resolution file is provided (e.g. resolution1015.graphml), the regions are output as nodes in a NetworkX graph.

    Example
    -------

    >>> import nipype.interfaces.cmtk as cmtk
    >>> regval = cmtk.RegionalValues()
    >>> regval.inputs.in_file = 'pet_resliced.nii'
    >>> regval.inputs.segmentation_file = 'ROI_scale500.nii.gz'
    >>> regval.run() # doctest: +SKIP
    """
    input_spec = RegionalValuesInputSpec
    output_spec = RegionalValuesOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.lookup_table):
            LUT_dict = get_names(self.inputs.lookup_table)
                    
        if len(self.inputs.in_files) > 1:
            iflogger.info('Multiple input images detected')
            iflogger.info(len(self.inputs.in_files))
            in_files = self.inputs.in_files
        elif isdefined(self.inputs.in_file4d):
            iflogger.info('Single four-dimensional image selected')
            in_files = nb.four_to_three(self.inputs.in_file4d)
        else:
            iflogger.info('Single functional image provided')
            in_files = self.inputs.in_files

        per_file_stats = {}
        seg_file = nb.load(self.inputs.segmentation_file)
        segmentationdata = seg_file.get_data()

        if isdefined(self.inputs.resolution_network_file):
            try:
                gp = nx.read_gpickle(self.inputs.resolution_network_file)
            except:
                gp = nx.read_graphml(self.inputs.resolution_network_file)
            nodedict = gp.node[gp.nodes()[0]]
            if not nodedict.has_key('dn_position'):
                iflogger.info("Creating node positions from segmentation")
                G = nx.Graph()
                for u, d in gp.nodes_iter(data=True):
                    iflogger.info('Node ID {id}'.format(id=int(u)))
                    G.add_node(int(u), d)
                    xyz = tuple(
                        np.mean(np.where(np.flipud(segmentationdata) == int(d["dn_correspondence_id"])), axis=1))
                    G.node[int(u)]['dn_position'] = xyz
                ntwkname = op.abspath('nodepositions.pck')
                nx.write_gpickle(G, ntwkname)
            else:
                ntwkname = self.inputs.resolution_network_file

            rois = get_roi_list(self.inputs.segmentation_file)
            roi_mean_tc, roi_max_tc, roi_min_tc, roi_std_tc, voxels = get_timecourse_by_region(
                in_files, self.inputs.segmentation_file, rois)

            stats = {}
            stats['func_max'] = roi_max_tc
            stats['func_mean'] = roi_mean_tc
            stats['func_min'] = roi_min_tc
            stats['func_stdev'] = roi_std_tc
            stats['number_of_voxels'] = voxels
            stats['rois'] = rois
            if isdefined(self.inputs.lookup_table):
                stats['roi_names'] = []
                for x in rois:
                    try:
                        stats['roi_names'].append(LUT_dict[x])
                    except KeyError:
                        stats['roi_names'].append("Unknown_ROI_" + str(x))

            global all_ntwks
            all_ntwks = list()
            for in_file_idx, in_file in enumerate(in_files):
                ntwks = list()
                per_file_stats = {}
                per_file_stats['func_max'] = roi_max_tc[:, in_file_idx]
                per_file_stats['func_mean'] = roi_mean_tc[:, in_file_idx]
                per_file_stats['func_min'] = roi_min_tc[:, in_file_idx]
                per_file_stats['func_stdev'] = roi_std_tc[:, in_file_idx]
                per_file_stats['rois'] = rois
                if isdefined(self.inputs.lookup_table):
                    per_file_stats['roi_names'] = [LUT_dict[x] for x in rois]
                #per_file_stats['number_of_voxels'] = voxels[in_file_idx]
                for key in per_file_stats.keys():
                    iflogger.info(key)
                    iflogger.info(np.shape(per_file_stats[key]))
                    try:
                        nwtk = nx.read_gpickle(ntwkname)
                        newntwk = add_node_data(per_file_stats[key], nwtk)
                    except:
                        iflogger.error(key)
                        iflogger.error(np.shape(per_file_stats[key]))
                        iflogger.error(
                            "Double-check the subjects' freesurfer directory.")

                        raise Exception(
                            "There may not be enough regions in the segmentation file!")

                    name = '{k}_{i}'.format(k=key, i=str(in_file_idx))
                    out_file = op.abspath(name + '.pck')
                    nx.write_gpickle(newntwk, out_file)
                    ntwks.append(out_file)
                all_ntwks.extend(ntwks)
        else:
            rois = get_roi_list(self.inputs.segmentation_file)
            roi_mean_tc, roi_max_tc, roi_min_tc, roi_std_tc, voxels = get_timecourse_by_region(
                in_files, self.inputs.segmentation_file, rois)

            stats = {}
            stats['func_max'] = roi_max_tc
            stats['func_mean'] = roi_mean_tc
            stats['func_min'] = roi_min_tc
            stats['func_stdev'] = roi_std_tc
            stats['number_of_voxels'] = voxels
            stats['rois'] = rois
            if isdefined(self.inputs.lookup_table):
                stats['roi_names'] = []
                for x in rois:
                    try:
                        stats['roi_names'].append(LUT_dict[x])
                    except KeyError:
                        stats['roi_names'].append("Unknown_ROI_" + str(x))

        if isdefined(self.inputs.subject_id):
            stats['subject_id'] = self.inputs.subject_id

        out_stats_file = op.abspath(self.inputs.out_stats_file)
        iflogger.info(
            'Saving image statistics as {stats}'.format(stats=out_stats_file))
        sio.savemat(out_stats_file, stats)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        out_stats_file = op.abspath(self.inputs.out_stats_file)
        outputs["stats_file"] = out_stats_file
        if isdefined(self.inputs.resolution_network_file):
            outputs["networks"] = all_ntwks
        else:
            outputs["networks"] = ''
        return outputs

        def _gen_outfilename(self, name, ext):
            return name + '.' + ext


def tinv(P, df):
    """
    This function reproduces 'tinv' from MATLAB

    Inverse of Student's T cumulative distribution function (cdf).
    X=TINV(P,V) returns the inverse of Student's T cdf with V degrees
    of freedom, at the values in P.
    """
    from scipy.stats import t
    rv = t(df, 0)
    t_inverse = rv.ppf(P)
    return t_inverse


def get_roi_values(roi, segmentationdata, in_files):
    roi_means = []
    roi_maxs = []
    roi_mins = []
    roi_stds = []
    x, y, z = np.where(segmentationdata == roi)
    if not len(x) == 0:
        for in_file_idx, in_file in enumerate(in_files):
            if isinstance(in_file, str):
                functional = nb.load(in_file)
                functionaldata = functional.get_data()
            else:
                functionaldata = in_file.get_data()
            assert np.shape(segmentationdata)[0:3] == np.shape(functionaldata)[0:3]
            roi_mean = np.mean(functionaldata[x, y, z])
            roi_means.append(roi_mean)
            roi_max = np.max(functionaldata[x, y, z])
            roi_maxs.append(roi_max)
            roi_min = np.min(functionaldata[x, y, z])
            roi_mins.append(roi_min)
            roi_std = np.std(functionaldata[x, y, z])
            roi_stds.append(roi_std)
    else:
        for in_file_idx, in_file in enumerate(in_files):
            roi_means.append(0)
            roi_mins.append(0)
            roi_maxs.append(0)
            roi_stds.append(0)
    roi_means = np.array(roi_means)
    roi_maxs = np.array(roi_maxs)
    roi_mins = np.array(roi_mins)
    roi_stds = np.array(roi_stds)
    voxels = np.array(len(x))
    return roi_means, roi_maxs, roi_mins, roi_stds, voxels


def get_timecourse_by_region(in_files, segmentation_file, rois):
    roi_mean_tc = []
    roi_min_tc = []
    roi_max_tc = []
    roi_std_tc = []
    iflogger.info('Segmentation image: {img}'.format(img=segmentation_file))
    segmentation_file = nb.load(segmentation_file)
    segmentationdata = segmentation_file.get_data()
    iflogger.info('Found {roi} unique region values'.format(roi=len(rois)))
    voxels = []
    for idx, roi in enumerate(rois):
        iflogger.info('Region ID: {i}'.format(i=roi))
        roi_means, roi_maxs, roi_mins, roi_stds, voxels = get_roi_values(
            roi, segmentationdata, in_files)
        if idx == 0:
            roi_mean_tc = roi_means
            roi_max_tc = roi_maxs
            roi_min_tc = roi_mins
            roi_std_tc = roi_stds
            voxel_list = voxels
        else:
            roi_mean_tc = np.vstack((roi_mean_tc, roi_means))
            roi_max_tc = np.vstack((roi_max_tc, roi_maxs))
            roi_min_tc = np.vstack((roi_min_tc, roi_mins))
            roi_std_tc = np.vstack((roi_std_tc, roi_stds))
            voxel_list = np.vstack((voxel_list, voxels))
    voxel_list = np.array(voxel_list)
    print voxel_list
    return roi_mean_tc, roi_max_tc, roi_min_tc, roi_std_tc, voxel_list


class SimpleTimeCourseCorrelationGraphInputSpec(TraitedSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True, xor=[
                              'in_file4d'], desc='Original functional magnetic resonance image (fMRI) as a set of 3-dimensional images')
    in_file4d = File(exists=True, mandatory=True, xor=[
                     'in_files'], desc='Original functional magnetic resonance image (fMRI) as a 4-dimension image')
    segmentation_file = File(exists=True, mandatory=True,
                             desc='Image with segmented regions (e.g. aparc+aseg.nii or the output from cmtk.Parcellate())')
    structural_network = File(exists=True, mandatory=True,
                              desc='Structural connectivity network, built from white-matter tracts and the input segmentation file.')
    out_network_file = File('simplecorrelation.pck', usedefault=True,
                            desc='The output functional network as a NetworkX gpickle (.pck)')
    out_stats_file = File('stats.mat', usedefault=True,
                          desc='Some simple image statistics for regions saved as a Matlab .mat')


class SimpleTimeCourseCorrelationGraphOutputSpec(TraitedSpec):
    network_file = File(
        desc='Output gpickled network files for all statistical measures')
    stats_file = File(
        desc='Some simple image statistics for the original and normalized images saved as a Matlab .mat')


class SimpleTimeCourseCorrelationGraph(BaseInterface):
    input_spec = SimpleTimeCourseCorrelationGraphInputSpec
    output_spec = SimpleTimeCourseCorrelationGraphOutputSpec

    def _run_interface(self, runtime):
        if len(self.inputs.in_files) > 1:
            iflogger.info('Multiple input images detected')
            iflogger.info(len(self.inputs.in_files))
            in_files = self.inputs.in_files
        elif isdefined(self.inputs.in_file4d):
            iflogger.info('Single four-dimensional image selected')
            in_file4d = nb.load(self.inputs.in_file4d)
            in_files = nb.four_to_three(in_file4d)
        else:
            iflogger.info('Single functional image provided')
            in_files = self.inputs.in_files

        if len(in_files) == 1:
            iflogger.error(
                "Only one functional image was input. Pearson's correlation coefficient can not be calculated")
            raise ValueError
        else:
            rois = get_roi_list(self.inputs.segmentation_file)
            fMRI_timecourse = get_timecourse_by_region(
                in_files, self.inputs.segmentation_file, rois)

        timecourse_at_each_node = fMRI_timecourse.T
        iflogger.info(np.shape(timecourse_at_each_node))
        iflogger.info(
            'Structural Network: {s}'.format(s=self.inputs.structural_network))
        structural_network = nx.read_gpickle(self.inputs.structural_network)
        rois = structural_network.nodes()
        #rois = get_roi_list(self.inputs.segmentation_file)
        number_of_nodes = len(rois)
        iflogger.info('Found {roi} unique region values'.format(roi=len(rois)))

        newntwk = structural_network.copy()
        newntwk = remove_all_edges(newntwk)

        simple_correlation_matrix = np.zeros(
            (number_of_nodes, number_of_nodes))
        iflogger.info('Drawing edges...')
        for idx_i, roi_i in enumerate(rois):
            #iflogger.info('ROI:{i}, T-value: {t}'.format(i=roi_i, t=t_i))
            for idx_j, roi_j in enumerate(rois):
                #iflogger.info('...ROI:{j}, T-value: {t}'.format(j=roi_j, t=t_j))
                if idx_j > idx_i:
                    simple_correlation_matrix[idx_i, idx_j] = pearsonr(
                        fMRI_timecourse[idx_i], fMRI_timecourse[idx_j])[0]
                elif roi_i == roi_j:
                    simple_correlation_matrix[idx_i, idx_j] = 0.5

        simple_correlation_matrix = simple_correlation_matrix + \
            simple_correlation_matrix.T
        stats = {'correlation': simple_correlation_matrix}
        newntwk = add_edge_data(simple_correlation_matrix, newntwk)
        path, name, ext = split_filename(self.inputs.out_network_file)
        if not ext == '.pck':
            ext = '.pck'
        out_network_file = op.abspath(name + ext)
        iflogger.info(
            'Saving simple correlation network as {out}'.format(out=out_network_file))
        nx.write_gpickle(newntwk, out_network_file)

        path, name, ext = split_filename(self.inputs.out_stats_file)
        if not ext == '.mat':
            ext = '.mat'
        out_stats_file = op.abspath(name + ext)

        iflogger.info(
            'Saving image statistics as {stats}'.format(stats=out_stats_file))
        sio.savemat(out_stats_file, stats)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        path, name, ext = split_filename(self.inputs.out_stats_file)
        if not ext == '.mat':
            ext = '.mat'
        out_stats_file = op.abspath(name + ext)
        outputs["stats_file"] = out_stats_file
        path, name, ext = split_filename(self.inputs.out_network_file)
        if not ext == '.pck':
            ext = '.pck'
        out_network_file = op.abspath(name + ext)
        outputs["network_file"] = out_network_file
        return outputs

    def _gen_outfilename(self, name, ext):
        return name + '.' + ext
