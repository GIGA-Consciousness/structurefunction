import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl
import nipype.interfaces.mrtrix as mrtrix

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from coma.helpers import (add_subj_name_to_sfmask, add_subj_name_to_wmmask,
                          add_subj_name_to_T1, add_subj_name_to_T1brain,
                          add_subj_name_to_termmask, add_subj_name_to_aparc,
                          select_ribbon, select_GM, select_WM, select_CSF,
                          wm_labels_only)


def anatomically_constrained_tracking(name="fiber_tracking", lmax=4):
    '''
    Define inputs and outputs of the workflow
    '''
    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=["subject_id",
                                                 "dwi",
                                                 "bvecs",
                                                 "bvals",
                                                 "single_fiber_mask",
                                                 "wm_mask",
                                                 "termination_mask",
                                                 "registration_matrix_file",
                                                 "registration_image_file",
                                                 ]),
        name="inputnode")

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=["fiber_odfs",
                                                 "fiber_tracks_tck_dwi",
                                                 "fiber_tracks_trk_t1"]),
        name="outputnode")

    '''
    Define the nodes
    '''

    fsl2mrtrix = pe.Node(interface=mrtrix.FSL2MRTrix(), name='fsl2mrtrix')
    fsl2mrtrix.inputs.invert_y = True

    estimateresponse = pe.Node(interface=mrtrix.EstimateResponseForSH(),
                               name='estimateresponse')

    estimateresponse.inputs.maximum_harmonic_order = lmax

    csdeconv = pe.Node(interface=mrtrix.ConstrainedSphericalDeconvolution(),
                       name='csdeconv')
    csdeconv.inputs.maximum_harmonic_order = lmax

    CSDstreamtrack = pe.Node(
        interface=mrtrix.ProbabilisticSphericallyDeconvolutedStreamlineTrack(
        ),
        name='CSDstreamtrack')

    CSDstreamtrack.inputs.desired_number_of_tracks = 100000
    CSDstreamtrack.inputs.minimum_tract_length = 10

    tck2trk = pe.Node(interface=mrtrix.MRTrix2TrackVis(), name='tck2trk')

    '''
    Connect the workflow
    '''
    workflow = pe.Workflow(name=name)
    workflow.base_dir = name

    '''
    Structural processing to create seed and termination masks
    '''

    # Might be worthwhile to use a smaller file? Could be faster to load but only
    # the dimensions of the "image_file" are using in this interface
    workflow.connect([(inputnode, fsl2mrtrix, [("bvecs", "bvec_file"),
                                               ("bvals", "bval_file")])])

    workflow.connect([(inputnode, tck2trk,[("dwi","image_file")])])

    workflow.connect([(inputnode, tck2trk,[("registration_image_file","registration_image_file")])])
    workflow.connect([(inputnode, tck2trk,[("registration_matrix_file","matrix_file")])])

    workflow.connect([(inputnode, estimateresponse, [("single_fiber_mask", "mask_image")])])

    workflow.connect([(inputnode, estimateresponse, [("dwi", "in_file")])])
    workflow.connect([(fsl2mrtrix, estimateresponse, [("encoding_file", "encoding_file")])])

    workflow.connect([(inputnode, csdeconv, [("dwi", "in_file")])])

    #workflow.connect([(inputnode, csdeconv, [("termination_mask", "mask_image")])])

    workflow.connect(
        [(estimateresponse, csdeconv, [("response", "response_file")])])
    workflow.connect(
        [(fsl2mrtrix, csdeconv, [("encoding_file", "encoding_file")])])
    workflow.connect(
        [(inputnode, CSDstreamtrack, [("wm_mask", "seed_file")])])
    workflow.connect(
        [(inputnode, CSDstreamtrack, [("termination_mask", "mask_file")])])
    workflow.connect(
        [(csdeconv, CSDstreamtrack, [("spherical_harmonics_image", "in_file")])])

    workflow.connect([(CSDstreamtrack, tck2trk, [("tracked", "in_file")])])

    workflow.connect([
         (CSDstreamtrack, outputnode, [("tracked", "fiber_tracks_tck_dwi")]),
         (csdeconv, outputnode, [("spherical_harmonics_image", "fiber_odfs")]),
         (tck2trk, outputnode, [("out_file", "fiber_tracks_trk_t1")]),
         ])

    return workflow
