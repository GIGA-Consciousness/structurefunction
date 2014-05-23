import os
import os.path as op
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.mrtrix as mrtrix
import nipype.interfaces.cmtk as cmtk
from nipype.workflows.misc.utils import select_aparc

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from coma.interfaces import RegionalValues, nonlinfit_fn, CMR_glucose

def dmn_labels_only(in_file, out_filename):
    from nipype.utils.filemanip import split_filename
    import nibabel as nb
    import numpy as np
    import os.path as op
    in_image = nb.load(in_file)
    in_header = in_image.get_header()
    in_data = in_image.get_data()

    out_data = np.zeros(np.shape(in_data))
    out_data[np.where(in_data==2)] = 1
    out_data[np.where(in_data==41)] = 1
    out_data[np.where(in_data==82)] = 1
    out_data[np.where(in_data==251)] = 1
    out_data[np.where(in_data==252)] = 1
    out_data[np.where(in_data==253)] = 1
    out_data[np.where(in_data==254)] = 1
    out_data[np.where(in_data==255)] = 1
    out_data[np.where(in_data==49)] = 1
    out_data[np.where(in_data==10)] = 1


    _, name, _ = split_filename(in_file)
    out_file = op.abspath(out_filename)
    try:
        out_image = nb.Nifti1Image(
            data=out_data, header=in_header, affine=in_image.get_affine())
    except TypeError:
        out_image = nb.Nifti1Image(
            dataobj=out_data, header=in_header, affine=in_image.get_affine())
    nb.save(out_image, out_file)
    return out_file


def get_rois(roi_file):
    import nibabel as nb
    import numpy as np
    roi_image = nb.load(roi_file)
    roi_data = roi_image.get_data()
    rois = np.unique(roi_data)
    rois = rois.tolist()
    rois.remove(0)
    rois.sort()
    return rois

def split_roi(roi_file):
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
    for roi in rois:
        new_data = roi_data.copy()
        new_data[roi_data==roi] = 1
        new_data[roi_data!=roi] = 0
        new_image = nb.Nifti1Image(dataobj=new_data, affine=roi_image.get_affine(),
            header=roi_image.get_header())
        new_image.set_data_dtype(np.uint8)
        out_file = op.abspath("Region_ID_%s.nii.gz" % roi)
        nb.save(new_image, out_file)
        print("Written to %s" % out_file)
        roi_files.append(out_file)
    return roi_files

def combine_rois(rois_to_combine):
    import os.path as op
    import nibabel as nb
    import numpy as np
    import string
    import random
    from nipype.utils.filemanip import split_filename
    print("Combining %s" ",".join(rois_to_combine))
    image = nb.load(rois_to_combine[0])
    new_data = np.zeros(np.shape(image.get_data()))
    ids = []

    for roi_file in rois_to_combine:
        _, roi_name, _ = split_filename(roi_file)
        roi_name = roi_name.replace("Region_ID_","")
        roi_name = roi_name.replace(".","-")
        ids.append(roi_name)
        roi_image = nb.load(roi_file)
        roi_data = roi_image.get_data()
        new_data = new_data + roi_data

    new_data[new_data > 0] = 1
    new_data[new_data < 0] = 0

    new_image = nb.Nifti1Image(dataobj=new_data, affine=image.get_affine(),
        header=image.get_header())
    out_file = op.abspath("MergedRegions_%s.nii.gz" % "_".join(ids))
    nb.save(new_image, out_file)
    print("Written to %s" % out_file)
    return roi_file

def inclusion_filtering(track_file, roi_file, fa_file, md_file):
    import os.path as op
    import nibabel as nb
    import numpy as np
    import glob
    from coma.workflows.dmn import split_roi, get_rois, combine_rois
    import nipype.pipeline.engine as pe
    import nipype.interfaces.fsl as fsl
    import nipype.interfaces.mrtrix as mrtrix

    rois = get_rois(roi_file)
    roi_files = split_roi(roi_file)
    out_files = []
    for roi_i in rois:
        for roi_j in rois:
            if roi_j > roi_i:
                idpair = "%0.1f_%0.1f" % (roi_i, roi_j)
                idpair = idpair.replace(".","-")
                roi_i_file = [s for s in roi_files if "%0.1f" % roi_i in s]
                roi_j_file = [s for s in roi_files if "%0.1f" % roi_j in s]
                rois_to_combine = [roi_i_file[0], roi_j_file[0]]
                inclusion_mask = combine_rois(rois_to_combine)

                filter_tracks = pe.Node(interface=mrtrix.FilterTracks(), name='filt_%s' % idpair)
                filter_tracks.inputs.in_file = track_file
                filter_tracks.inputs.include_file = inclusion_mask
                filter_tracks.out_filename = op.abspath("FiltTracks_%s.tck" % idpair)

                tracks2tdi = pe.Node(interface=mrtrix.Tracks2Prob(), name='tdi_%s' % idpair)
                tracks2tdi.inputs.template_file = fa_file
                tracks2tdi.inputs.out_filename = op.abspath("TDI_%s.nii.gz" % idpair)
                tracks2tdi.inputs.output_datatype = "Int16"

                binarize_tdi_mask_fa = pe.Node(interface=fsl.MultiImageMaths(), name='binarize_tdi_mask_fa_%s' % idpair)
                binarize_tdi_mask_fa.inputs.op_string = "-bin -mul %s"
                binarize_tdi_mask_fa.inputs.operand_files = [fa_file]
                binarize_tdi_mask_fa.inputs.out_file = "FA_%s.nii.gz" % idpair

                binarize_tdi_mask_md = binarize_tdi_mask_fa.clone(name='binarize_tdi_mask_md_%s' % idpair)
                binarize_tdi_mask_md.inputs.operand_files = [md_file]
                binarize_tdi_mask_md.inputs.out_file = "MD_%s.nii.gz" % idpair

                mean_fa = pe.Node(interface=fsl.ImageStats(op_string = '-M'), name = 'mean_fa_%s' % idpair) 
                mean_md = pe.Node(interface=fsl.ImageStats(op_string = '-M'), name = 'mean_md_%s' % idpair) 

                workflow = pe.Workflow(name=idpair)
                workflow.base_dir = op.abspath(idpair)

                workflow.connect(
                    [(filter_tracks, tracks2tdi, [("out_file", "in_file")])])
                workflow.connect(
                    [(tracks2tdi, binarize_tdi_mask_fa, [("tract_image", "in_file")])])
                workflow.connect(
                    [(tracks2tdi, binarize_tdi_mask_md, [("tract_image", "in_file")])])
                workflow.connect(
                    [(binarize_tdi_mask_fa, mean_fa, [("out_file", "in_file")])])
                workflow.connect(
                    [(binarize_tdi_mask_md, mean_md, [("out_file", "in_file")])])

                workflow.config['execution'] = {'remove_unnecessary_outputs': 'false',
                                                   'hash_method': 'timestamp'}

                workflow.run()
                md_masked = glob.glob(op.abspath(op.join(idpair,idpair,'binarize_tdi_mask_md_%s' % idpair, "MD_%s.nii.gz" % idpair))[0])
                fa_masked = glob.glob(op.abspath(op.join(idpair,idpair,'binarize_tdi_mask_fa_%s' % idpair, "FA_%s.nii.gz" % idpair))[0])
                tracks    = glob.glob(op.abspath(op.join(idpair,idpair,'filt_%s' % idpair, "FiltTracks_%s.tck" % idpair))[0])
                out_files.append(md_masked)
                out_files.append(fa_masked)
                out_files.append(tracks)
                #workflow.nodes
                print(out_files)
                #node1.result.outputs
                # Generate TDI
                # Binarize TDI
                # Multiply by FA and MD
                # fslstats mean FA and MD
                # Add to "connectivity matrix"

    return out_files


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
        interface=util.IdentityInterface(fields=["track_file",
                                                 "fa",
                                                 "md",
                                                 "roi_file"]),
        name="inputnode")




    binarize_rois = pe.Node(interface=fsl.ImageMaths(), name='binarize_rois')
    binarize_rois.inputs.op_string = "-bin"

    filter_tracks = pe.Node(interface=mrtrix.FilterTracks(), name='filter_tracks')

    incl_filt_interface = util.Function(input_names=["track_file", "roi_file", "fa_file", "md_file"],
        output_names=["out_file"], function=inclusion_filtering)
    paired_inclusion_filtering = pe.Node(interface=incl_filt_interface, name='paired_inclusion_filtering')

    output_fields = ["fa"]
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
        [(inputnode, paired_inclusion_filtering, [("roi_file", "roi_file")])])
    workflow.connect(
        [(inputnode, paired_inclusion_filtering, [("fa", "fa_file")])])
    workflow.connect(
        [(inputnode, paired_inclusion_filtering, [("md", "md_file")])])
    return workflow

