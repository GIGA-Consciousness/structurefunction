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

fsl.FSLCommand.set_default_output_type('NIFTI')

from coma.interfaces import RegionalValues, nonlinfit_fn, CMR_glucose

def split_roi(roi_file):
    import os.path as op
    import nibabel as nb
    import numpy as np
    roi_image = nb.load(roi_file)
    roi_data = roi_image.get_data()
    rois = np.unique(roi_data)
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
    from nipype.filemanip.utils import split_filename

    image = nb.load(rois_to_combine[0])
    new_data = np.zeros(np.shape(image.get_data()))
    ids = []

    for roi_file in rois_to_combine:
        _, roi_name, _ = split_filename(roi_file)
        ids.append(roi_name.replace("Region_ID_",""))
        roi_image = nb.load(roi_file)
        roi_data = roi_image.get_data()
        new_data = new_data + roi_data

    new_data[new_data > 0] = 1
    new_data[new_data < 0] = 0

    out_file = op.abspath("MergedRegions_%s.nii.gz" % "_".join(ids))
    nb.save(new_image, out_file)
    print("Written to %s" % out_file)
    return roi_file

def inclusion_filtering(track_file, roi_file, fa_file, md_file):
    import os.path as op
    import nibabel as nb
    import numpy as np

    roi_files = split_roi(roi_file)

    for roi_i in rois:
        for roi_j in rois:
            if roi_j > roi_i:
                idpair = "%s_%s" % (roi_i, roi_j)
                rois_to_combine = [roi_i, roi_j]
                inclusion_mask = combine_rois(rois_to_combine)

                filter_tracks = pe.Node(interface=mrtrix.FilterTracks(), name='filt_%s' % idpair)
                filter_tracks.inputs.in_file = track_file
                filter_tracks.inputs.roi_file = inclusion_mask
                filter_tracks.out_filename = op.abspath("FiltTracks_%s.tck" % idpair)

                tracks2tdi = pe.Node(interface=mrtrix.Tracks2Prob(), name='tdi_%s' % idpair)
                tracks2tdi.inputs.template_file = fa_file

                binarize_tdi_mask_fa = pe.Node(interface=fsl.MultiImageMaths(), name='binarize_tdi_mask_fa_%s' % idpair)
                binarize_tdi_mask_fa.inputs.op_string = "-bin -mul %s"

                binarize_tdi_mask_md = binarize_tdi_mask_fa.clone(name='binarize_tdi_mask_md_%s' % idpair)

                mean_fa = pe.Node(interface=fsl.ImageStats(op_string = '-M'), name = 'mean_fa_%s' % idpair) 
                mean_md = pe.Node(interface=fsl.ImageStats(op_string = '-M'), name = 'mean_fa_%s' % idpair) 

                workflow = pe.Workflow(name=idpair)
                workflow.base_output_dir = idpair

                workflow.connect(
                    [(filter_tracks, tracks2tdi, [("out_file", "in_file")])])
                workflow.connect(
                    [(tracks2tdi, binarize_tdi_mask_fa, [("out_file", "in_file")])])
                workflow.connect(
                    [(tracks2tdi, binarize_tdi_mask_md, [("out_file", "in_file")])])

                workflow.run()
                #node1 = G.nodes()[0]
                #node1.result.outputs
                # Generate TDI
                # Binarize TDI
                # Multiply by FA and MD
                # fslstats mean FA and MD
                # Add to "connectivity matrix"

    out_file = op.abspath("Test.npz")
    return out_file


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
        [(binarize_rois, filter_tracks, [("out_file", "in_file")])])
    workflow.connect(
        [(filter_tracks, paired_inclusion_filtering, [("out_file", "in_file")])])
    workflow.connect(
        [(inputnode, paired_inclusion_filtering, [("roi_file", "in_file")])])
    workflow.connect(
        [(inputnode, paired_inclusion_filtering, [("fa", "fa_file")])])
    workflow.connect(
        [(inputnode, paired_inclusion_filtering, [("md", "md_file")])])
    return workflow

