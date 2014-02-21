def CMR_glucose(in_file, dose, weight, delay, glycemie, scan_time=15):
    '''
    Scales an image by the calculated cerebral metabolic rate of glucose

        in_file: Nifti image (.nii, .nii.gz, or .img/.hdr)
        dose: milliCurie (mCi)
        weight: kilograms (kg)
        delay: minutes (min)
        glycemie: milligrams / deciliter (mg/dL)
        scan_time: minutes (min)
    '''
    import os
    import os.path as op
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import split_filename

    T = delay + scan_time
    k1 = 0.087
    k2 = 0.203
    k3 = 0.127
    lumped = 0.8

    # Convert dose to megabecquerels (MBq)
    dose_mbq = dose * 37.0

    std_txt_path = op.join(os.environ["COMA_DIR"], 'etc', 'standard_cmrglc.txt')
    standard = np.loadtxt(std_txt_path)
    t1 = standard[:, 0] / 60.0
    ti = np.arange(t1[0], 90, 0.1)

    from scipy.interpolate import InterpolatedUnivariateSpline
    spline = InterpolatedUnivariateSpline(t1, standard[:,1])
    ca_y = spline(ti)

    t = ti[ti < T]

    ca = ca_y[ti < T] * 1000 * dose_mbq / weight

    cax = ca * np.exp(-(k2 + k3) * (T - t))

    integral = np.trapz(cax, t)
    mecalc = k1 * integral

    auc = np.trapz(ca, t)

    denom = auc - mecalc / k1
    Ca = glycemie / 18.0
    CAdivlumped = Ca / lumped
    cax2 = CAdivlumped

    _, name, _ = split_filename(in_file)
    image = nb.load(in_file)
    data = image.get_data()

    slope = cax2 / denom
    inter = -cax2 * mecalc / denom

    rescaled = data*slope + inter
    rescaled_image = nb.Nifti1Image(data=rescaled, affine=image.get_affine())
    out_file = op.abspath(name + '_CMRGLC2.nii.gz')

    nb.save(rescaled_image, out_file)
    return out_file