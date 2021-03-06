def scale_PVC_matrix_fn(subject_id, in_file, dose, weight, delay, scan_time=15, isotope="F18", height=None, glycemie=None, scale_SUV_by_glycemia=True):
    '''
    Scales the NumPy results to calculate SUV and 

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
    import scipy.io as sio

    # Load matrix
    pvc = np.load(in_file)

    T = delay + scan_time
    k1 = 0.087
    k2 = 0.203
    k3 = 0.127
    lumped = 0.8

    # Convert dose to megabecquerels (MBq)
    dose_mbq = dose * 37.0

    std_txt_path = op.join(
        os.environ["COMA_DIR"], 'etc', 'standard_cmrglc.txt')
    standard = np.loadtxt(std_txt_path)
    t1 = standard[:, 0] / 60.0
    ti = np.arange(t1[0], 90, 0.1)

    from scipy.interpolate import InterpolatedUnivariateSpline
    spline = InterpolatedUnivariateSpline(t1, standard[:, 1])
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

    slope = cax2 / denom
    inter = -cax2 * mecalc / denom


    # Half-life is in seconds
    if isotope == "F18":
        half_life = 109.8 * 60
    elif isotope == 'C11':
        half_life = 20.334 * 60
    elif isotope == 'O15':
        half_life = 122.24
    elif isotope == 'N13':
        half_life = 9.97 * 60

    methods = pvc.keys()
    new_dict = {}
    new_dict["subject_id"] = subject_id
    new_dict["scale_SUV_by_glycemia"] = scale_SUV_by_glycemia
    new_dict["isotope"] = isotope
    new_dict["half_life"] = half_life
    new_dict["weight"] = weight
    new_dict["assumed_AIF"] = standard
    new_dict["delay"] = delay
    new_dict["dose_mCi"] = dose
    new_dict["dose_MBq"] = dose_mbq
    new_dict["glycemia"] = glycemie

    if height is not None:
        body_surface_area = 0.007184 * \
            (weight ^ 0.425 * height ^ 0.725)  # DuBois formula
        new_dict["height"] = height
        new_dict["body_surface_area"] = body_surface_area

    try:
        extra_fields = ["gm_file", "wm_slice_used",
            "pet_file", "region_names", "VOLUMES_(cc)"]
        for extra in extra_fields:
            new_dict[extra] = pvc[extra]
            methods.remove(extra)

    except ValueError:
        print('NPZ data does not have expected keys')
    
    for method in methods:
        data = pvc[method]
        new_dict["orig_" + method] = data
        
        AIF = data * slope + inter
        new_dict["AIF_"+ method] = AIF

        if height is not None:
            SUV = data * np.power(2, (T / half_life)) / \
                (dose_mbq / body_surface_area)
        else:
            SUV = data * np.power(2, (T / half_life)) / \
                (dose_mbq / (weight / 1000.))
        if scale_SUV_by_glycemia:
            SUV = SUV / glycemie
        
        new_dict["SUV_"+ method] = SUV

    out_matlab_mat = op.abspath(subject_id + "_PET_AIF_SUV_data.mat")
    out_npz = op.abspath(subject_id + "_PET_AIF_SUV_data.npz")

    sio.savemat(out_matlab_mat, new_dict)
    np.savez(out_npz, **new_dict)
    return out_npz, out_matlab_mat



def CMR_glucose(subject_id, in_file, dose, weight, delay, glycemie, scan_time=15):
    '''
    Scales an image to the calculated cerebral metabolic rate of glucose
    using a standard arterial input curve

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

    T = delay + scan_time
    k1 = 0.087
    k2 = 0.203
    k3 = 0.127
    lumped = 0.8

    # Convert dose to megabecquerels (MBq)
    dose_mbq = dose * 37.0

    std_txt_path = op.join(
        os.environ["COMA_DIR"], 'etc', 'standard_cmrglc.txt')
    standard = np.loadtxt(std_txt_path)
    t1 = standard[:, 0] / 60.0
    ti = np.arange(t1[0], 90, 0.1)

    from scipy.interpolate import InterpolatedUnivariateSpline
    spline = InterpolatedUnivariateSpline(t1, standard[:, 1])
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

    image = nb.load(in_file)
    data = image.get_data()

    slope = cax2 / denom
    inter = -cax2 * mecalc / denom

    rescaled = data * slope + inter
    rescaled_image = nb.Nifti1Image(
        dataobj=rescaled, affine=image.get_affine())
    out_file = op.abspath(subject_id + '_CMRGLC2.nii.gz')
    nb.save(rescaled_image, out_file)
    return out_file, cax2, mecalc, denom


def calculate_SUV(subject_id, in_file, dose, weight, delay, scan_time=15, isotope="F18", height=None, glycemie=None):
    '''
    Calculates standardized uptake value

        in_file: Nifti image (.nii, .nii.gz, or .img/.hdr)
        dose: milliCurie (mCi)
        weight: kilograms (kg)
        delay: minutes (min)
        scan_time: minutes (min)
        isotope: the radiotracer isotope. Must be one of "F18", "C11", "O15", or "N13".
        height: metres (optional, returns body body surface area SUV if provided)
    '''
    import os.path as op
    import numpy as np
    import nibabel as nb

    image = nb.load(in_file)
    data = image.get_data()

    # Half-life is in seconds
    if isotope == "F18":
        half_life = 109.8 * 60
    elif isotope == 'C11':
        half_life = 20.334 * 60
    elif isotope == 'O15':
        half_life = 122.24
    elif isotope == 'N13':
        half_life = 9.97 * 60

    # Convert dose to megabecquerels (MBq)
    dose_mbq = dose * 37.0

    T = delay + scan_time

    if height is not None:
        body_surface_area = 0.007184 * \
            (weight ^ 0.425 * height ^ 0.725)  # DuBois formula
        SUV = data * np.power(2, (T / half_life)) / \
            (dose_mbq / body_surface_area)
    else:
        SUV = data * np.power(2, (T / half_life)) / \
            (dose_mbq / (weight / 1000.))

    if glycemie is not None:
        SUV = SUV / glycemie
    
    SUV_image = nb.Nifti1Image(dataobj=SUV, affine=image.get_affine())
    out_file = op.abspath(subject_id + '_SUV.nii.gz')
    nb.save(SUV_image, out_file)
    return out_file
