def inclusion_filtering_mrtrix3(track_file, roi_file, fa_file, md_file, roi_names=None, registration_image_file=None, registration_matrix_file=None, prefix=None, tdi_threshold=10):
    import os
    import os.path as op
    import numpy as np
    import glob
    from coma.workflows.dmn import get_rois, save_heatmap
    from coma.interfaces.dti import write_trackvis_scene
    import nipype.pipeline.engine as pe
    import nipype.interfaces.fsl as fsl
    import nipype.interfaces.mrtrix as mrtrix
    import nipype.interfaces.diffusion_toolkit as dtk
    from nipype.utils.filemanip import split_filename
    import subprocess
    import shutil

    rois = get_rois(roi_file)

    fa_out_matrix = op.abspath("%s_FA.csv" % prefix)
    md_out_matrix = op.abspath("%s_MD.csv" % prefix)
    invLen_invVol_out_matrix = op.abspath("%s_invLen_invVol.csv" % prefix)

    subprocess.call(["tck2connectome", "-assignment_voxel_lookup",
        "-zero_diagonal",
        "-metric", "mean_scalar", "-image", fa_file,
        track_file, roi_file, fa_out_matrix])

    subprocess.call(["tck2connectome", "-assignment_voxel_lookup",
        "-zero_diagonal",
        "-metric", "mean_scalar", "-image", md_file,
        track_file, roi_file, md_out_matrix])

    subprocess.call(["tck2connectome", "-assignment_voxel_lookup",
        "-zero_diagonal",
        "-metric", "invlength_invnodevolume",
        track_file, roi_file, invLen_invVol_out_matrix])

    subprocess.call(["tcknodeextract", "-assignment_voxel_lookup",
        track_file, roi_file, prefix + "_"])

    fa_matrix_thr = np.zeros((len(rois), len(rois)))
    md_matrix_thr = np.zeros((len(rois), len(rois)))
    tdi_matrix = np.zeros((len(rois), len(rois)))
    track_volume_matrix = np.zeros((len(rois), len(rois)))

    out_files = []
    track_files = []
    for idx_i, roi_i in enumerate(rois):
        for idx_j, roi_j in enumerate(rois):
            if idx_j >= idx_i:

                filtered_tracks = glob.glob(op.abspath(prefix + "_%s-%s.tck" % (roi_i, roi_j)))[0]
                print(filtered_tracks)

                if roi_names is None:
                    roi_i = str(int(roi_i))
                    roi_j = str(int(roi_j))
                    idpair = "%s_%s" % (roi_i, roi_j)
                    idpair = idpair.replace(".","-")
                else:
                    roi_name_i = roi_names[idx_i]
                    roi_name_j = roi_names[idx_j]
                    idpair = "%s_%s" % (roi_name_i, roi_name_j)

                tracks2tdi = pe.Node(interface=mrtrix.Tracks2Prob(), name='tdi_%s' % idpair)
                tracks2tdi.inputs.template_file = fa_file
                tracks2tdi.inputs.in_file = filtered_tracks
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

                tck2trk = mrtrix.MRTrix2TrackVis()
                tck2trk.inputs.image_file = fa_file
                tck2trk.inputs.in_file = filtered_tracks
                trk_file = op.abspath("%s_%s.trk" % (prefix, idpair))
                tck2trk.inputs.out_filename = trk_file
                tck2trk.base_dir = op.abspath(".")

                if registration_image_file is not None and registration_matrix_file is not None:
                    tck2trk.inputs.registration_image_file = registration_image_file
                    tck2trk.inputs.matrix_file = registration_matrix_file

                workflow = pe.Workflow(name=idpair)
                workflow.base_dir = op.abspath(idpair)

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
                tck2trk.run()

                fa_masked = glob.glob(out_fa_name)[0]
                md_masked = glob.glob(out_md_name)[0]

                if roi_names is not None:
                    tracks = op.abspath(prefix + "_%s-%s.tck" % (roi_name_i, roi_name_j))
                    shutil.move(filtered_tracks, tracks)
                else:
                    tracks = filtered_tracks

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

                if track_volume == 0:
                    os.remove(fa_masked)
                    os.remove(md_masked)
                    os.remove(tdi)
                else:
                    out_files.append(md_masked)
                    out_files.append(fa_masked)
                    out_files.append(tracks)
                    out_files.append(tdi)
                
                if op.exists(trk_file):
                    out_files.append(trk_file)
                    track_files.append(trk_file)

                assert(0 <= mean_fa < 1)
                fa_matrix_thr[idx_i, idx_j] = mean_fa
                md_matrix_thr[idx_i, idx_j] = mean_md
                tdi_matrix[idx_i, idx_j] = mean_tdi
                track_volume_matrix[idx_i, idx_j] = track_volume


    fa_matrix = np.loadtxt(fa_out_matrix)
    md_matrix = np.loadtxt(md_out_matrix)
    fa_matrix = fa_matrix + fa_matrix.T
    md_matrix = md_matrix + md_matrix.T
    fa_matrix_thr = fa_matrix_thr + fa_matrix_thr.T
    md_matrix_thr = md_matrix_thr + md_matrix_thr.T
    tdi_matrix = tdi_matrix + tdi_matrix.T

    invLen_invVol_matrix = np.loadtxt(invLen_invVol_out_matrix)
    invLen_invVol_matrix = invLen_invVol_matrix + invLen_invVol_matrix.T

    track_volume_matrix = track_volume_matrix + track_volume_matrix.T
    if prefix is not None:
        npz_data = op.abspath("%s_connectivity.npz" % prefix)
    else:
        _, prefix, _ = split_filename(track_file)
        npz_data = op.abspath("%s_connectivity.npz" % prefix)
    np.savez(npz_data, fa=fa_matrix, md=md_matrix, tdi=tdi_matrix, trkvol=track_volume_matrix,
        fa_thr=fa_matrix_thr, md_thr=md_matrix_thr, invLen_invVol=invLen_invVol_matrix)


    print("Saving heatmaps...")
    fa_heatmap = save_heatmap(fa_matrix, roi_names, '%s_fa' % prefix)
    fa_heatmap_thr = save_heatmap(fa_matrix_thr, roi_names, '%s_fa_thr' % prefix)
    md_heatmap = save_heatmap(md_matrix, roi_names, '%s_md' % prefix)
    md_heatmap_thr = save_heatmap(md_matrix_thr, roi_names, '%s_md_thr' % prefix)
    tdi_heatmap = save_heatmap(tdi_matrix, roi_names, '%s_tdi' % prefix)
    trk_vol_heatmap = save_heatmap(track_volume_matrix, roi_names, '%s_trk_vol' % prefix)

    invLen_invVol_heatmap = save_heatmap(invLen_invVol_matrix, roi_names, '%s_invLen_invVol' % prefix)
    
    
    summary_images = []
    summary_images.append(fa_heatmap)
    summary_images.append(fa_heatmap_thr)    
    summary_images.append(md_heatmap)
    summary_images.append(md_heatmap_thr)    
    summary_images.append(tdi_heatmap)
    summary_images.append(trk_vol_heatmap)
    summary_images.append(invLen_invVol_heatmap)


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