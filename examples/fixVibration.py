#!/usr/bin/env python
import os, sys
import string

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
    gradients = np.loadtxt(bvecs).T

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



def fixVib(in_dwi, in_bvec, in_bval, x_thresh=0.6):
	from nipype.utils.filemanip import split_filename
	x_thresh = float(x_thresh)
	_, name, _ = split_filename(in_dwi)
	print("Removing bad volumes...")
	corr_dwi, corr_bvec, corr_bval, n = remove_bad_volumes(in_dwi, in_bvec, in_bval, x_thresh)
	out_basename = name + "_Corr%0.3f" % x_thresh
	print("Fitting tensors...")
	nonlinfit_fn(corr_dwi, corr_bvec, corr_bval, out_basename)
	return out_basename


if __name__ == '__main__':
	# Should probably use argparse for this
	in_dwi, in_bvec, in_bval, x_thresh = sys.argv[1:]
	fixVib(in_dwi, in_bvec, in_bval, x_thresh)
