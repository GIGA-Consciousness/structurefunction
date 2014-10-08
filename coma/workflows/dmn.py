import os.path as op
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl
import nipype.interfaces.mrtrix as mrtrix
from coma.interfaces.mrtrix3 import inclusion_filtering_mrtrix3

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

def save_heatmap(in_array, labels, out_name, colormap=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from nipype.utils.filemanip import split_filename
    import seaborn as sns

    fig, ax = plt.subplots()
    if colormap is None:
        heatmap = ax.pcolor(in_array, cmap=plt.cm.Blues)
    else:
        heatmap = ax.pcolor(in_array, cmap=colormap)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(labels, minor=False)
    ax.set_yticklabels(labels, minor=False)
    cbar = plt.colorbar(heatmap)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(in_array.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(in_array.shape[1])+0.5, minor=False)

    #plt.xticks(rotation=45)

    path, name, _ = split_filename(out_name)
    out_file = op.join(path, name + ".pdf")
    plt.savefig(out_file)
    return out_file


def dmn_labels_combined(in_file, out_filename):
    from nipype.utils.filemanip import split_filename
    import nibabel as nb
    import numpy as np
    import os.path as op
    in_image = nb.load(in_file)
    in_header = in_image.get_header()
    in_data = in_image.get_data()

    out_data = np.zeros(np.shape(in_data))
    out_data = out_data.astype(int)

    # Thalami
    out_data[np.where(in_data==10)] = 1
    out_data[np.where(in_data==49)] = 2

    # Frontal
    # Medial orbitofrontal
    out_data[np.where(in_data==1014)] = 3
    out_data[np.where(in_data==2014)] = 4
    # Superior frontal
    out_data[np.where(in_data==1028)] = 3
    out_data[np.where(in_data==2028)] = 4
    # Rostral anterior cingulate
    out_data[np.where(in_data==1026)] = 3
    out_data[np.where(in_data==2026)] = 4

    # Parietal
    # Posterior cingulate
    out_data[np.where(in_data==1023)] = 5
    out_data[np.where(in_data==2023)] = 6
    # Precuneus
    out_data[np.where(in_data==1025)] = 5
    out_data[np.where(in_data==2025)] = 6
    # Isthmus 
    out_data[np.where(in_data==1010)] = 5
    out_data[np.where(in_data==2010)] = 6

    # Inferior parietal
    out_data[np.where(in_data==1008)] = 7
    out_data[np.where(in_data==2008)] = 8
    
    _, name, _ = split_filename(in_file)
    out_file = op.abspath(out_filename)
    try:
        out_image = nb.Nifti1Image(
            data=out_data, header=in_header, affine=in_image.get_affine())
    except TypeError:
        out_image = nb.Nifti1Image(
            dataobj=out_data, header=in_header, affine=in_image.get_affine())
    out_image.set_data_dtype(np.float32)
    nb.save(out_image, out_file)
    return out_file


def get_rois(roi_file):
    import nibabel as nb
    import numpy as np
    roi_image = nb.load(roi_file)
    roi_data = roi_image.get_data()
    roi_data = roi_data.astype(int)
    rois = np.unique(roi_data)
    rois = rois.tolist()
    rois.remove(0)
    rois.sort()
    return rois

def split_roi(roi_file, roi_names=None, prefix=None, binarize=True):
    import os.path as op
    import nibabel as nb
    import numpy as np
    roi_image = nb.load(roi_file)
    roi_data = roi_image.get_data()
    rois = np.unique(roi_data)
    rois = rois.tolist()
    rois.remove(0)
    rois.sort()

    roi_files = []
    for idx, roi in enumerate(rois):
        new_data = roi_data.copy()
        if binarize == True:
            new_data[roi_data==roi] = 1
        new_data[roi_data!=roi] = 0
        new_image = nb.Nifti1Image(dataobj=new_data, affine=roi_image.get_affine(),
            header=roi_image.get_header())
        new_image.set_data_dtype(np.uint8)

        if roi_names is None:
            roi_id = str(int(roi))
        else:
            roi_id = roi_names[idx]

        if prefix is None:
            out_file = op.abspath("Region_ID_%s.nii.gz" % roi_id)
        else:
            out_file = op.abspath("%s_Region_ID_%s.nii.gz" % (prefix, roi_id))

        nb.save(new_image, out_file)
        print("Written to %s" % out_file)
        roi_files.append(out_file)
    return roi_files


def inclusion_filtering(track_file, roi_file, fa_file, md_file, roi_names=None, registration_image_file=None, registration_matrix_file=None, prefix=None, tdi_threshold=10):
    import os
    import os.path as op
    import numpy as np
    import glob
    from coma.workflows.dmn import split_roi, get_rois, save_heatmap
    from coma.interfaces.dti import write_trackvis_scene
    import nipype.pipeline.engine as pe
    import nipype.interfaces.fsl as fsl
    import nipype.interfaces.mrtrix as mrtrix
    import nipype.interfaces.diffusion_toolkit as dtk
    from nipype.utils.filemanip import split_filename

    rois = get_rois(roi_file)
    roi_files = split_roi(roi_file, roi_names, prefix)
    out_files = []

    fa_matrix = np.zeros((len(rois), len(rois)))
    md_matrix = np.zeros((len(rois), len(rois)))
    tdi_matrix = np.zeros((len(rois), len(rois)))
    track_volume_matrix = np.zeros((len(rois), len(rois)))

    track_files = []
    for idx_i, roi_i in enumerate(rois):
        if roi_names is None:
            roi_i_file = [s for s in roi_files if "%d" % roi_i in s]
            roi_i = str(int(roi_i))
        else:
            roi_i_file = [s for s in roi_files if "%s" % roi_names[idx_i] in s]
            roi_i = roi_names[idx_i]

        filter_tracks_roi_i = pe.Node(interface=mrtrix.FilterTracks(), name='filt_%s' % roi_i)
        filter_tracks_roi_i.inputs.in_file = track_file
        filter_tracks_roi_i.inputs.include_file = roi_i_file[0]
        out_roi_i_filtered = op.abspath("%s_FiltTracks_%s.tck" % (prefix, roi_i))
        filter_tracks_roi_i.inputs.out_filename = out_roi_i_filtered
        filter_tracks_roi_i.run()

        for idx_j, roi_j in enumerate(rois):
            if idx_j > idx_i:
                if roi_names is None:
                    roi_j = str(int(roi_j))
                    idpair = "%s_%s" % (roi_i, roi_j)
                    idpair = idpair.replace(".","-")
                    roi_j_file = [s for s in roi_files if "%s" % roi_j in s]
                else:
                    roi_j = roi_names[idx_j]
                    roi_j_file = [s for s in roi_files if "%s" % roi_names[idx_j] in s]
                    idpair = "%s_%s" % (roi_i, roi_j)

                filter_tracks_roi_i_roi_j = pe.Node(interface=mrtrix.FilterTracks(), name='filt_%s' % idpair)
                filter_tracks_roi_i_roi_j.inputs.in_file = out_roi_i_filtered
                filter_tracks_roi_i_roi_j.inputs.include_file = roi_j_file[0]
                filter_tracks_roi_i_roi_j.inputs.out_filename = "%s_FiltTracks_%s.tck" % (prefix, idpair)

                tracks2tdi = pe.Node(interface=mrtrix.Tracks2Prob(), name='tdi_%s' % idpair)
                tracks2tdi.inputs.template_file = fa_file
                out_tdi_name = op.abspath("%s_TDI_%s.nii.gz" % (prefix, idpair))
                tracks2tdi.inputs.out_filename = out_tdi_name
                tracks2tdi.inputs.output_datatype = "Int16"

                binarize_tdi = pe.Node(interface=fsl.ImageMaths(), name='binarize_tdi_%s' % idpair)
                binarize_tdi.inputs.op_string = "-thr %d -bin" % tdi_threshold
                out_tdi_vol_name = op.abspath("%s_TDI_bin_%d_%s.nii.gz" % (prefix, tdi_threshold, idpair))
                binarize_tdi.inputs.out_file = out_tdi_vol_name

                mask_fa = pe.Node(interface=fsl.MultiImageMaths(), name='mask_fa_%s' % idpair)
                mask_fa.inputs.op_string = "-mul %s"
                mask_fa.inputs.operand_files = [fa_file]
                out_fa_name = op.abspath("%s_FA_%s.nii.gz" % (prefix, idpair))
                mask_fa.inputs.out_file = out_fa_name

                mask_md = mask_fa.clone(name='mask_md_%s' % idpair)
                mask_md.inputs.operand_files = [md_file]
                out_md_name = op.abspath("%s_MD_%s.nii.gz" % (prefix, idpair))
                mask_md.inputs.out_file = out_md_name

                mean_fa = pe.Node(interface=fsl.ImageStats(op_string = '-M'), name = 'mean_fa_%s' % idpair) 
                mean_md = pe.Node(interface=fsl.ImageStats(op_string = '-M'), name = 'mean_md_%s' % idpair)
                mean_tdi = pe.Node(interface=fsl.ImageStats(op_string = '-l %d -M' % tdi_threshold), name = 'mean_tdi_%s' % idpair)
                track_volume = pe.Node(interface=fsl.ImageStats(op_string = '-l %d -V' % tdi_threshold), name = 'track_volume_%s' % idpair)

                tck2trk = pe.Node(interface=mrtrix.MRTrix2TrackVis(), name='tck2trk')
                tck2trk.inputs.image_file = fa_file
                tck2trk.inputs.out_filename = idpair + ".trk"


                if registration_image_file is not None and registration_matrix_file is not None:
                    tck2trk.inputs.registration_image_file = registration_image_file
                    tck2trk.inputs.matrix_file = registration_matrix_file

                workflow = pe.Workflow(name=idpair)
                workflow.base_dir = op.abspath(idpair)

                workflow.connect(
                    [(filter_tracks_roi_i_roi_j, tracks2tdi, [("out_file", "in_file")])])
                workflow.connect(
                    [(filter_tracks_roi_i_roi_j, tck2trk, [("out_file", "in_file")])])
                workflow.connect(
                    [(tracks2tdi, binarize_tdi, [("tract_image", "in_file")])])
                workflow.connect(
                    [(binarize_tdi, mask_fa, [("out_file", "in_file")])])
                workflow.connect(
                    [(binarize_tdi, mask_md, [("out_file", "in_file")])])
                workflow.connect(
                    [(mask_fa, mean_fa, [("out_file", "in_file")])])
                workflow.connect(
                    [(mask_md, mean_md, [("out_file", "in_file")])])
                workflow.connect(
                    [(tracks2tdi, mean_tdi, [("tract_image", "in_file")])])
                workflow.connect(
                    [(tracks2tdi, track_volume, [("tract_image", "in_file")])])

                workflow.config['execution'] = {'remove_unnecessary_outputs': 'false',
                                                   'hash_method': 'timestamp'}

                result = workflow.run()
                fa_masked = glob.glob(out_fa_name)[0]
                md_masked = glob.glob(out_md_name)[0]
                tracks    = glob.glob(op.abspath(op.join(idpair,idpair,'filt_%s' % idpair, "%s_FiltTracks_%s.tck" % (prefix, idpair))))[0]
                tdi = glob.glob(out_tdi_vol_name)[0]

                nodes = result.nodes()
                node_names = [s.name for s in nodes]
                
                mean_fa_node = [nodes[idx] for idx, s in enumerate(node_names) if "mean_fa" in s][0]
                mean_fa = mean_fa_node.result.outputs.out_stat
                
                mean_md_node = [nodes[idx] for idx, s in enumerate(node_names) if "mean_md" in s][0]
                mean_md = mean_md_node.result.outputs.out_stat
                
                mean_tdi_node = [nodes[idx] for idx, s in enumerate(node_names) if "mean_tdi" in s][0]
                mean_tdi = mean_tdi_node.result.outputs.out_stat
                
                track_volume_node = [nodes[idx] for idx, s in enumerate(node_names) if "track_volume" in s][0]
                track_volume = track_volume_node.result.outputs.out_stat[1] # First value is in voxels, 2nd is in volume

                trk_node = [nodes[idx] for idx, s in enumerate(node_names) if "tck2trk" in s][0]
                trk_file = trk_node.result.outputs.out_file

                if track_volume == 0:
                    os.remove(fa_masked)
                    os.remove(md_masked)
                    os.remove(tdi)
                    os.remove(tracks)
                else:
                    out_files.append(md_masked)
                    out_files.append(fa_masked)
                    out_files.append(tracks)
                    out_files.append(tdi)
                    out_files.append(trk_file)

                track_files.append(trk_file)

                assert(0 <= mean_fa < 1)
                fa_matrix[idx_i, idx_j] = mean_fa
                md_matrix[idx_i, idx_j] = mean_md
                tdi_matrix[idx_i, idx_j] = mean_tdi
                track_volume_matrix[idx_i, idx_j] = track_volume


    fa_matrix = fa_matrix + fa_matrix.T
    md_matrix = md_matrix + md_matrix.T
    tdi_matrix = tdi_matrix + tdi_matrix.T
    track_volume_matrix = track_volume_matrix + track_volume_matrix.T
    if prefix is not None:
        npz_data = op.abspath("%s_connectivity.npz" % prefix)
    else:
        _, prefix, _ = split_filename(track_file)
        npz_data = op.abspath("%s_connectivity.npz" % prefix)
    np.savez(npz_data, fa=fa_matrix, md=md_matrix, tdi=tdi_matrix, trkvol=track_volume_matrix)


    print("Saving heatmaps...")
    fa_heatmap = save_heatmap(fa_matrix, roi_names, '%s_fa' % prefix)
    md_heatmap = save_heatmap(md_matrix, roi_names, '%s_md' % prefix)
    tdi_heatmap = save_heatmap(tdi_matrix, roi_names, '%s_tdi' % prefix)
    trk_vol_heatmap = save_heatmap(track_volume_matrix, roi_names, '%s_trk_vol' % prefix)
    
    summary_images = []
    summary_images.append(fa_heatmap)
    summary_images.append(md_heatmap)
    summary_images.append(tdi_heatmap)
    summary_images.append(trk_vol_heatmap)


    out_merged_file = op.abspath('%s_MergedTracks.trk' % prefix)
    skip = 80.
    track_merge = pe.Node(interface=dtk.TrackMerge(), name='track_merge')
    track_merge.inputs.track_files = track_files
    track_merge.inputs.output_file = out_merged_file
    track_merge.run()

    track_names = []
    for t in track_files:
        _, name, _ = split_filename(t)
        track_names.append(name)

    out_scene = op.abspath("%s_MergedScene.scene" % prefix)
    out_scene_file = write_trackvis_scene(out_merged_file, n_clusters=len(track_files), skip=skip, names=track_names, out_file=out_scene)
    print("Merged track file written to %s" % out_merged_file)
    print("Scene file written to %s" % out_scene_file)
    out_files.append(out_merged_file)
    out_files.append(out_scene_file)
    return out_files, npz_data, summary_images


def create_paired_tract_analysis_wf(name="track_filtering"):
    '''
    Generate whole brain tracks
    Filter tracks using regions of DMN
    Then Creatematrix
    Filter tracks by pairs of regions
    FA multiply by binarized track mask
    Mean FA or MD in tract per tract
    Mean FAs into "connectivity" matrix.
    '''

    '''
    Define the nodes
    '''
    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=["subject_id",
                                                 "track_file",
                                                 "fa",
                                                 "md",
                                                 "roi_file",
                                                 "roi_names",
                                                 "registration_image_file",
                                                 "registration_matrix_file"]),
        name="inputnode")




    binarize_rois = pe.Node(interface=fsl.ImageMaths(), name='binarize_rois')
    binarize_rois.inputs.op_string = "-bin"

    filter_tracks = pe.Node(interface=mrtrix.FilterTracks(), name='filter_tracks')

    incl_filt_interface = util.Function(input_names=["track_file", "roi_file", "fa_file", "md_file",
        "roi_names", "registration_image_file", "registration_matrix_file", "prefix", "tdi_threshold"],
        output_names=["out_files", "npz_data", "summary_images"], function=inclusion_filtering_mrtrix3)
    paired_inclusion_filtering = pe.Node(interface=incl_filt_interface, name='paired_inclusion_filtering')

    output_fields = ["connectivity_files", "connectivity_data", "summary_images"]
    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=output_fields),
        name="outputnode")

    '''
    Set up the workflow connections
    '''
    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    workflow.connect(
        [(inputnode, filter_tracks, [("track_file", "in_file")])])
    workflow.connect(
        [(inputnode, binarize_rois, [("roi_file", "in_file")])])
    workflow.connect(
        [(binarize_rois, filter_tracks, [("out_file", "include_file")])])
    workflow.connect(
        [(filter_tracks, paired_inclusion_filtering, [("out_file", "track_file")])])
    workflow.connect(
        [(inputnode, paired_inclusion_filtering, [("roi_names", "roi_names")])])
    workflow.connect(
        [(inputnode, paired_inclusion_filtering, [("subject_id", "prefix")])])
    workflow.connect(
        [(inputnode, paired_inclusion_filtering, [("roi_file", "roi_file")])])
    workflow.connect(
        [(inputnode, paired_inclusion_filtering, [("fa", "fa_file")])])
    workflow.connect(
        [(inputnode, paired_inclusion_filtering, [("md", "md_file")])])
    workflow.connect(
        [(inputnode, paired_inclusion_filtering, [("registration_image_file", "registration_image_file")])])
    workflow.connect(
        [(inputnode, paired_inclusion_filtering, [("registration_matrix_file", "registration_matrix_file")])])
    workflow.connect(
        [(paired_inclusion_filtering, outputnode, [("out_files", "connectivity_files")])])
    workflow.connect(
        [(paired_inclusion_filtering, outputnode, [("npz_data", "connectivity_data")])])
    workflow.connect(
        [(paired_inclusion_filtering, outputnode, [("summary_images", "summary_images")])])
    return workflow

