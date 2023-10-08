import os

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

brats_dir = "/home/matthis/datasets/BraTS_2021/"
image_dir = "BraTS2021_00002"

path = brats_dir+image_dir
if not os.path.exists("../images"):
    os.mkdir("../images")
mod_list = ["t1", "t2", "t1ce", "flair", "seg"]
count=0
for image_dir in os.listdir(brats_dir):
    if count==8:
        break
    path = brats_dir + image_dir
    img = nib.load(os.path.join(path, image_dir + "_t1ce.nii.gz")).get_fdata()
    seg = nib.load(os.path.join(path, image_dir + "_seg.nii.gz")).get_fdata()
    if seg[:,:,80].sum() == 0:
        count+=1
        plt.imshow(img[:, :, 80].transpose(), cmap="gray")
        plt.axis("off")
        plt.savefig("../images/" + image_dir + ".png", bbox_inches='tight')
        plt.show()
    """for mod in mod_list:
        img = nib.load(os.path.join(path, image_dir + "_" + mod + ".nii.gz")).get_fdata()
        if mod == "seg":
            img_rgb = np.zeros((240,240,155,3))
            img_rgb[img == 1] = [255, 255,0]
            img_rgb[img == 2] = [255, 0, 0]
            img_rgb[img == 4] = [0, 0, 255]
            plt.imshow(np.transpose(img_rgb[:,:,80],(1,0,2)))
        else:
            plt.imshow(img[:,:,80].transpose(), cmap="gray")
        plt.axis("off")
        plt.savefig("../images/" + mod + ".png",bbox_inches='tight')
        plt.show()"""
