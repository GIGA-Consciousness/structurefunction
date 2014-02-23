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

    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)

    MD = dti.mean_diffusivity(tenfit.evals)
    norm = dti.norm(tenfit.quadratic_form)

    RGB = color_fa(FA, tenfit.evecs)

    evecs = tenfit.evecs.astype(np.float32)

    evals = tenfit.evals.astype(np.float32)

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