from coma.interfaces.dti import remove_bad_volumes, nonlinfit_fn

in_dwi = "KayaDoke20120903_134740DTI64dirs029a001_motion_eddy.nii.gz"
in_bvec = "KayaDoke20120903_134740DTI64dirs029a001_rotated.bvec"
in_bval = "KayaDoke20120903_134740DTI64dirs029a001.bval"
x_thresh = 0.55

corr_dwi, corr_bvec, corr_bval, n = remove_bad_volumes(in_dwi, in_bvec, in_bval, x_thresh)
nonlinfit_fn(corr_dwi, corr_bvec, corr_bval, "Corrected")