import os
import torch
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import random

from models.HUnet import UNet, HUNetv4
from dataset_brats import TestBrats
from loss_func import TotalLoss
from tools.metrics import random_crop_test


def plot_attention(source, attention):
    img = source.squeeze().detach().cpu().numpy()[:,:,80]
    img = (img - img.min())/(img.max() - img.min()) * 255
    img = np.uint8(np.stack([img, img, img], axis=2))
    att = attention.squeeze().detach().cpu().numpy()[:,:,80]
    att = (att - att.min())/(att.max() - att.min()) * 255
    att= np.uint8(att)
    heatmap_img = cv2.applyColorMap(att, cv2.COLORMAP_JET)
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(super_imposed_img.transpose((1,0,2)))
    ax[1].imshow(img.transpose((1,0,2)))
    plt.show()


path_baseline = "baseline_2018_t1ce_full_fold1_0126_1318"
path_student = "student_2018_t1ce_full_fold1_0201_0051"
res_dir = os.path.join(os.environ["HOME"], "Nextcloud/knowledge-distillation/result/")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
checkpoint_name = 'Checkpoint_best_fold_1.pkl'
checkpoint_path_baseline = os.path.join(res_dir, path_baseline, 'pkl', checkpoint_name)
checkpoint_path_student = os.path.join(res_dir, path_student, 'pkl', checkpoint_name)
checkpoint_baseline = torch.load(checkpoint_path_baseline)
checkpoint_student = torch.load(checkpoint_path_student)
config_baseline = checkpoint_baseline["config"]
config_student = checkpoint_student["config"]
config_student['data_split_dir'] = os.path.join(os.environ["HOME"], "Nextcloud/knowledge-distillation/result/", path_student, "data_split")
config_baseline['data_split_dir'] = os.path.join(os.environ["HOME"], "Nextcloud/knowledge-distillation/result/", path_baseline, "data_split")
config_student["device"] = device
config_baseline["device"] = device
net_baseline = UNet(config_baseline).to(device)
net_student = HUNetv4(config_student).to(device)
net_baseline.load_state_dict(checkpoint_baseline["model_state_dict"])
net_student.load_state_dict(checkpoint_student["model_state_dict"])
valid_set_file = os.path.join(res_dir, path_baseline, 'data_split', 'valid_set_fold_1.txt')
with open(valid_set_file, "r") as f:
    valid_set = f.readlines()
    valid_set = valid_set[0].split(',')
    f.close()
valid_set = ["/home/matthis/Nextcloud/" + "/".join(file.split('/')[-5:]) for file in valid_set]
dataset = TestBrats(config_student, valid_set)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10)
criterion = TotalLoss(config_student)
net_student.eval()
net_baseline.eval()
progress_bar = tqdm(total=len(test_loader))
with torch.no_grad():
    dice_et = []
    dice_tc = []
    dice_wt = []
    for i, data in enumerate(test_loader, 0):
        if config_student['debug_mode']:
            if i >= 5:
                break
        imgs, labels = data
        inputs = [img.to(config_student['device']) for img in imgs]
        labels = labels.to(config_student['device'])

        student_pred = random_crop_test(net_student, inputs, "target", config_student)
        baseline_pred = random_crop_test(net_baseline, inputs, "target", config_baseline)

        loss_dict, dice_student = criterion(student_pred, labels, similarity=False)
        loss_dict, dice_baseline = criterion(baseline_pred, labels, similarity=False)
        dice_et.append((dice_baseline['Dice_ET'], dice_student['Dice_ET']))
        dice_tc.append((dice_baseline['Dice_TC'], dice_student['Dice_TC']))
        dice_wt.append((dice_baseline['Dice_WT'], dice_student['Dice_WT']))
        progress_bar.update(1)
    dice_et = np.array(dice_et)
    dice_tc = np.array(dice_tc)
    dice_wt = np.array(dice_wt)
    np.save("dice_et.npy", dice_et)
    np.save("dice_tc.npy", dice_tc)
    np.save("dice_wt.npy", dice_wt)
    progress_bar.close()

