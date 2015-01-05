def dmn_labels_combined(in_file, out_filename):
    from nipype.utils.filemanip import split_filename
    import nibabel as nb
    import numpy as np
    import os.path as op
    in_image = nb.load(in_file)
    in_header = in_image.get_header()
    in_data = in_image.get_data()

    out_data = np.zeros(np.shape(in_data))
    out_data = out_data.astype(int)

    ## This is the PET DTI option (uncomment this)
    # Thalami
    out_data[np.where(in_data==10)] = 1
    out_data[np.where(in_data==49)] = 2

    # Frontal
    # Medial orbitofrontal
    out_data[np.where(in_data==1014)] = 3
    out_data[np.where(in_data==2014)] = 4
    # Superior frontal
    out_data[np.where(in_data==1028)] = 3
    out_data[np.where(in_data==2028)] = 4
    # Rostral anterior cingulate
    out_data[np.where(in_data==1026)] = 3
    out_data[np.where(in_data==2026)] = 4

    # Parietal
    # Posterior cingulate
    out_data[np.where(in_data==1023)] = 5
    out_data[np.where(in_data==2023)] = 6
    # Precuneus
    out_data[np.where(in_data==1025)] = 5
    out_data[np.where(in_data==2025)] = 6
    # Isthmus 
    out_data[np.where(in_data==1010)] = 5
    out_data[np.where(in_data==2010)] = 6

    # Inferior parietal
    out_data[np.where(in_data==1008)] = 7
    out_data[np.where(in_data==2008)] = 8

     ## This is the Actigait option (uncomment this)
    # Thalami
    #out_data[np.where(in_data==10)] = 1
    #out_data[np.where(in_data==49)] = 2

    # Frontal
    # Medial orbitofrontal
    #out_data[np.where(in_data==1014)] = 3
    #out_data[np.where(in_data==2014)] = 4
    # Superior frontal
    #out_data[np.where(in_data==1028)] = 3
    #out_data[np.where(in_data==2028)] = 4
    # Frontal pole
    #out_data[np.where(in_data==1032)] = 3
    #out_data[np.where(in_data==2032)] = 4

    # Precentral
    #out_data[np.where(in_data==1024)] = 5
    #out_data[np.where(in_data==3024)] = 6
    
    # Postcentral
    #out_data[np.where(in_data==1022)] = 7
    #out_data[np.where(in_data==3022)] = 8
    
    # Paracentral
    #out_data[np.where(in_data==1017)] = 9
    #out_data[np.where(in_data==2017)] = 10
    


    _, name, _ = split_filename(in_file)
    out_file = op.abspath(out_filename)
    try:
        out_image = nb.Nifti1Image(
            data=out_data, header=in_header, affine=in_image.get_affine())
    except TypeError:
        out_image = nb.Nifti1Image(
            dataobj=out_data, header=in_header, affine=in_image.get_affine())
    out_image.set_data_dtype(np.float32)
    nb.save(out_image, out_file)
    return out_file