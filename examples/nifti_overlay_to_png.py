import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nb

def two_images_to_png(anat, overlay, out_file):

    anat_img = nb.load(anat)
    anat_data = anat_img.get_data()

    overlay_img = nb.load(overlay)
    overlay_data = overlay_img.get_data()
    
    x_slice = np.shape(anat_data)[0]/2
    y_slice = np.shape(anat_data)[1]/2
    z_slice = np.shape(anat_data)[2]/2
    x_slice = np.shape(overlay_data)[0]/2
    y_slice = np.shape(overlay_data)[1]/2
    z_slice = np.shape(overlay_data)[2]/2

    overlay_cmap = "jet"
    alpha = 0.5
    thresh = 0
    overlay_data[overlay_data <= thresh] = np.nan

    anat_x = anat_data[x_slice, :, :]
    overlay_x = overlay_data[x_slice, :, :]
    anat_y = anat_data[:, y_slice, :]
    overlay_y = overlay_data[:, y_slice, :]
    anat_z = anat_data[:, :, z_slice]
    overlay_z = overlay_data[:, :, z_slice]

    fig = plt.figure(figsize=(8,8),dpi=300,facecolor='black')

    a=fig.add_subplot(1,3,1, axisbg='k')
    # X-scale
    plt.imshow(anat_x, interpolation='nearest', cmap="Greys_r", origin="upper")
    plt.imshow(overlay_x, interpolation='nearest', cmap=overlay_cmap, alpha=alpha, origin="upper")
    plt.axis('off')

    a=fig.add_subplot(1,3,2)
    # Y-scale
    plt.imshow(anat_y, interpolation='nearest', cmap="Greys_r", origin="upper")
    plt.imshow(overlay_y, interpolation='nearest', cmap=overlay_cmap, alpha=alpha, origin="lower")

    a=fig.add_subplot(1,3,3)
    # Z-scale
    plt.imshow(anat_z, interpolation='nearest', cmap="Greys_r", origin="upper")
    plt.imshow(overlay_z, interpolation='nearest', cmap=overlay_cmap, alpha=alpha, origin="lower")

    #plt.show()
    plt.savefig(out_file, frameon=False, dpi=300, bbox_inches='tight', pad_inches=0,
        facecolor=fig.get_facecolor())
    return out_file

anat = op.abspath("ComaSample/data/Bend1/Bend1_T1.nii.gz")
overlay = op.abspath("ComaSample/data/Bend1/Bend1_term_mask.nii.gz")
out_file = two_images_to_png(anat, overlay, "Bend1_T1_PET.png")