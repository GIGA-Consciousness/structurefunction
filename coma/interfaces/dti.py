import random

def nonlinfit_fn(dwi, bvecs, bvals, base_name):
    import nibabel as nb
    import numpy as np
    import os.path as op
    import dipy.reconst.dti as dti
    from dipy.core.gradients import GradientTable

    dwi_img = nb.load(dwi)
    dwi_data = dwi_img.get_data()
    dwi_affine = dwi_img.get_affine()
    
    from dipy.segment.mask import median_otsu
    b0_mask, mask = median_otsu(dwi_data, 2, 4)
    # Mask the data so that tensors are not fit for
    # unnecessary voxels
    mask_img = nb.Nifti1Image(mask.astype(np.float32), dwi_affine)
    b0_imgs = nb.Nifti1Image(b0_mask.astype(np.float32), dwi_affine)
    b0_img = nb.four_to_three(b0_imgs)[0]

    out_mask_name = op.abspath(base_name + '_binary_mask.nii.gz')
    out_b0_name = op.abspath(base_name + '_b0_mask.nii.gz')
    nb.save(mask_img, out_mask_name)
    nb.save(b0_img, out_b0_name)

    # Load the gradient strengths and directions
    bvals = np.loadtxt(bvals)
    gradients = np.loadtxt(bvecs)

    # Dipy wants Nx3 arrays
    if gradients.shape[0] == 3:
        gradients = gradients.T
        assert(gradients.shape[1] == 3)

    # Place in Dipy's preferred format
    gtab = GradientTable(gradients)
    gtab.bvals = bvals

    # Fit the tensors to the data
    tenmodel = dti.TensorModel(gtab, fit_method="NLLS")
    tenfit = tenmodel.fit(dwi_data, mask)

    # Calculate the fit, fa, and md of each voxel's tensor
    tensor_data = tenfit.lower_triangular()
    print('Computing anisotropy measures (FA, MD, RGB)')
    from dipy.reconst.dti import fractional_anisotropy, color_fa

    evals = tenfit.evals.astype(np.float32)
    FA = fractional_anisotropy(np.abs(evals))
    FA = np.clip(FA, 0, 1)

    MD = dti.mean_diffusivity(np.abs(evals))
    norm = dti.norm(tenfit.quadratic_form)

    RGB = color_fa(FA, tenfit.evecs)

    evecs = tenfit.evecs.astype(np.float32)
    mode = tenfit.mode.astype(np.float32)
    mode = np.nan_to_num(mode)


    # Write tensor as a 4D Nifti image with the original affine
    tensor_fit_img = nb.Nifti1Image(tensor_data.astype(np.float32), dwi_affine)
    mode_img = nb.Nifti1Image(mode.astype(np.float32), dwi_affine)
    norm_img = nb.Nifti1Image(norm.astype(np.float32), dwi_affine)
    FA_img = nb.Nifti1Image(FA.astype(np.float32), dwi_affine)
    evecs_img = nb.Nifti1Image(evecs, dwi_affine)
    evals_img = nb.Nifti1Image(evals, dwi_affine)
    rgb_img = nb.Nifti1Image(np.array(255 * RGB, 'uint8'), dwi_affine)
    MD_img = nb.Nifti1Image(MD.astype(np.float32), dwi_affine)

    out_tensor_file = op.abspath(base_name + "_tensor.nii.gz")
    out_mode_file = op.abspath(base_name + "_mode.nii.gz")
    out_fa_file = op.abspath(base_name + "_fa.nii.gz")
    out_norm_file = op.abspath(base_name + "_norm.nii.gz")
    out_evals_file = op.abspath(base_name + "_evals.nii.gz")
    out_evecs_file = op.abspath(base_name + "_evecs.nii.gz")
    out_rgb_fa_file = op.abspath(base_name + "_rgb_fa.nii.gz")
    out_md_file = op.abspath(base_name + "_md.nii.gz")

    nb.save(rgb_img, out_rgb_fa_file)
    nb.save(norm_img, out_norm_file)
    nb.save(mode_img, out_mode_file)
    nb.save(tensor_fit_img, out_tensor_file)
    nb.save(evecs_img, out_evecs_file)
    nb.save(evals_img, out_evals_file)
    nb.save(FA_img, out_fa_file)
    nb.save(MD_img, out_md_file)
    print('Tensor fit image saved as {i}'.format(i=out_tensor_file))
    print('FA image saved as {i}'.format(i=out_fa_file))
    print('MD image saved as {i}'.format(i=out_md_file))
    return out_tensor_file, out_fa_file, out_md_file, \
        out_evecs_file, out_evals_file, out_rgb_fa_file, out_norm_file, \
        out_mode_file, out_mask_name, out_b0_name

def remove_bad_volumes(dwi, bvec_file, bval_file, thresh=0.8):
    import numpy as np
    import nibabel as nb
    import os.path as op
    from nipype.utils.filemanip import split_filename

    dwi_4D = nb.load(dwi)
    dwi_files = nb.four_to_three(dwi_4D)

    bvecs = np.transpose(np.loadtxt(bvec_file))
    bvals = np.transpose(np.loadtxt(bval_file))

    bad_indices = np.where(np.abs(bvecs[:,0]) >= thresh)[0]
    n_removed = len(bad_indices)

    dwi_files = [i for j, i in enumerate(dwi_files) if j not in bad_indices]
    bvecs = [i for j, i in enumerate(bvecs) if j not in bad_indices]
    bvals = [i for j, i in enumerate(bvals) if j not in bad_indices]

    corr_dwi = nb.concat_images(dwi_files)
    corr_bvecs = np.transpose(bvecs)
    corr_bvals = np.transpose(bvals)

    assert(len(dwi_files) == len(bvecs) == len(bvals))

    _, name, _ = split_filename(dwi)
    out_dwi = op.abspath(name + "_vib.nii.gz")

    _, bvec_name, _ = split_filename(bvec_file)
    out_bvecs = op.abspath(bvec_name + "_vib.bvec")

    _, bval_name, _ = split_filename(bval_file)
    out_bvals = op.abspath(bval_name + "_vib.bval")

    nb.save(corr_dwi, out_dwi)
    np.savetxt(out_bvecs, corr_bvecs)
    np.savetxt(out_bvals, corr_bvals)
    print("%d volumes were removed at threshold %f" % (n_removed, thresh))
    return out_dwi, out_bvecs, out_bvals, n_removed


# I forget where this function came from?
#https://gist.github.com/adewes/5884820
def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]
 
def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])
 
def generate_new_color(existing_colors,pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0,100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color

def hsv_to_rgb(h, s, v):
    """Converts HSV value to RGB values
    Hue is in range 0-359 (degrees), value/saturation are in range 0-1 (float)

    Direct implementation of:
    http://en.wikipedia.org/wiki/HSL_and_HSV#Conversion_from_HSV_to_RGB
    """
    h, s, v = [float(x) for x in (h, s, v)]

    hi = (h / 60) % 6
    hi = int(round(hi))

    f = (h / 60) - (h / 60)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    if hi == 0:
        return v, t, p
    elif hi == 1:
        return q, v, p
    elif hi == 2:
        return p, v, t
    elif hi == 3:
        return p, q, v
    elif hi == 4:
        return t, p, v
    elif hi == 5:
        return v, p, q

def write_trackvis_scene(track_file, n_clusters=1, skip=80, names=None, out_file = "NewScene.scene"):
    from random import randint, uniform
    bg_r, bg_g, bg_b = 0, 0, 0
    f = open(out_file, 'w')

    # Write some comments
    #f.write('<?xml version="1.0"?>\n')
    #f.write('<TrackVis>\n')
    f.write('<Comment>\n')
    f.write('    Scene file for TrackVis. Copyright (c) Ruopeng wang and Van J. Wedeen\n')
    f.write('    DO NOT mess with this file unless you know what you are doing.\n')
    f.write('      * Starting from version 2.0, all the parameters are represented in LPS coordinate.\n')
    f.write('        So x/y/z represent L/P/S. TrackVis will make neccessary transformation when loading\n')
    f.write('        older scene files.\n')
    f.write('    Written using coma.workflows.dti.write_trackvis_scene()\n')
    f.write('</Comment>\n')

    # Define the scene dimensions
    import nibabel.trackvis as trk
    _, hdr = trk.read(track_file)
    x, y, z = hdr['dim']
    vx, vy, vz = hdr['voxel_size']

    f.write('<Scene version="2.2">\n')
    f.write('    <Dimension x="%d" y="%d" z="%d" />\n' % (x, y, z))
    f.write('    <VoxelSize x="%f" y="%f" z="%f" />\n' % (vx, vy, vz))
    f.write('    <VoxelOrder current="LPS" original="LAS" />\n')
    f.write('    <LengthUnit value="0" />\n')

    from nipype.utils.filemanip import split_filename
    _, name, ext = split_filename(track_file)
    rpath = name + ext

    f.write('    <TrackFile path="%s" rpath="%s" />\n' % (track_file, rpath))
    f.write('    <Tracks>\n')
    
    colors = []
       
    for n in range(0,n_clusters):
        color = generate_new_color(colors,pastel_factor = 0.7)
        colors.append(color)
        h = randint(0, 300) # Select random green'ish hue from hue wheel
        s = uniform(0.2, 1)
        v = uniform(0.2, 1)

        r, g, b = hsv_to_rgb(h, s, v)
        r = randint(0,255)
        g = randint(0,255)
        b = randint(0,255)

        if names is not None:
            f.write('        <Track name="%s" id="%d">\n' % (str(names[n]), n+1000))
        else:
            f.write('        <Track name="Track %d" id="%d">\n' % (n,n+1000))

        f.write('            <Length low="0" high="1e+08" />\n')
        f.write('            <Property id="0" low="%d" high="%d" />\n' % (n,n)) # Determines which bundle is shown
        f.write('            <Slice plane="0" number="91" thickness="1" testmode="0" enable="0" visible="1" operator="and" id="1925142528"/>\n')
        f.write('            <Slice plane="1" number="109" thickness="1" testmode="0" enable="0" visible="1" operator="and" id="1881842394"/>\n')
        f.write('            <Slice plane="2" number="91" thickness="1" testmode="0" enable="0" visible="1" operator="and" id="2133446589"/>\n')
        f.write('            <Skip value="%f" enable="1" />\n' % skip)
        f.write('            <ShadingMode value="0" />\n')
        f.write('            <Radius value="0.05" />\n')
        f.write('            <NumberOfSides value="5" />\n')
        f.write('            <ColorCode value="1" />\n')
        f.write('            <SolidColor r="%d" g="%d" b="%d" />\n' % (r,g,b))
        f.write('            <ScalarIndex value="0" />\n')
        f.write('            <ScalarGradient>\n')
        f.write('                <ColorStop stop="0" r="1" g="1" b="0" />\n')
        f.write('                <ColorStop stop="1" r="1" g="0" b="0" />\n')
        f.write('            </ScalarGradient>\n')
        f.write('            <Saturation value="1" />\n')
        f.write('            <HelixPoint x="91" y="109" z="91" />\n')
        f.write('            <HelixVector x="1" y="0" z="0" />\n')
        f.write('            <HelixAxis visibility="1" />\n')
        f.write('            <Visibility value="1" />\n')
        f.write('            <AnnotationPosition x="67.8772" y="113.876" z="95.828" />\n')
        f.write('    </Track>\n')

    f.write('    <CurrentIndex value="0" />\n')
    f.write('</Tracks>\n')

    f.write('<LookUpTable>\n')
    f.write('    <DirectionScheme value="2" />\n')
    f.write('    <DirectionVector value="0" />\n')
    f.write('</LookUpTable>\n')
    f.write('<Coordinate>\n')
    f.write('    <Point000 x="0" y="0" z="0" id="-1" />\n')
    f.write('    <Point100 x="1" y="0" z="0" id="-1" />\n')
    f.write('    <Point##0 x="0" y="1" z="0" id="-1" />\n')
    f.write('    <Annotation type="1" />\n')
    f.write('</Coordinate>\n')
    f.write('<Camera>\n')
    f.write('    <Position x="91" y="-388.794" z="91" />\n')
    f.write('    <FocalPoint x="91" y="109" z="91" />\n')
    f.write('    <ViewUp x="0" y="2.28382e-16" z="1" />\n')
    f.write('    <ViewAngle value="30" />\n')
    f.write('    <ClippingRange near="333.159" far="702.527" />\n')
    f.write('</Camera>\n')
    f.write('<ObjectAnnotation value="0" />\n')
    f.write('<BackgroundColor r="%d" g="%d" b="%d" />\n' % (bg_r, bg_g, bg_b))
    f.write('</Scene>\n')
    return out_file

def bundle_tracks(in_file, dist_thr=40., pts = 16, skip=80.):
    import subprocess
    import os.path as op
    from nibabel import trackvis as tv
    from dipy.segment.quickbundles import QuickBundles
    streams, hdr = tv.read(in_file)
    streamlines = [i[0] for i in streams]
    qb = QuickBundles(streamlines, float(dist_thr), int(pts))
    clusters = qb.clustering
    #scalars = [i[0] for i in streams]

    out_files = []
    name = "quickbundle_"
    n_clusters = clusters.keys()
    print("%d clusters found" % len(n_clusters))

    new_hdr = tv.empty_header()
    new_hdr['n_scalars'] = 1

    for cluster in clusters:
        cluster_trk = op.abspath(name + str(cluster) + ".trk")
        print("Writing cluster %d to %s" % (cluster, cluster_trk))
        out_files.append(cluster_trk)
        clust_idxs = clusters[cluster]['indices']
        new_streams =  [ streamlines[i] for i in clust_idxs ]
        for_save = [(sl, None, None) for sl in new_streams]
        tv.write(cluster_trk, for_save, hdr)
    
    out_merged_file = "MergedBundles.trk"
    command_list = ["track_merge"]
    command_list.extend(out_files)
    command_list.append(out_merged_file)
    subprocess.call(command_list)
    out_scene_file = write_trackvis_scene(out_merged_file, n_clusters=len(clusters), skip=skip, names=None, out_file = "NewScene.scene")
    print("Merged track file written to %s" % out_merged_file)
    print("Scene file written to %s" % out_scene_file)
    return out_files, out_merged_file, out_scene_file
