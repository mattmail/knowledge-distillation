import os
import torch
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import random

from models.HUnet import UNet, HUNetv4
from dataset_brats import TestBrats, prepare_data
from loss_func import TotalLoss
from tools.metrics import random_crop_test
from scipy.signal import convolve2d

def plot_attention(source, attention, slice, labels):
    img = source.squeeze().detach().cpu().numpy()[:,:,slice]
    img = (img - img.min())/(img.max() - img.min()) * 255
    img = np.uint8(np.stack([img, img, img], axis=2))
    attention = attention.abs().sum(dim=1)
    att = attention.squeeze().detach().cpu().numpy()[:,:,slice]
    att = (att - att.min())/(att.max() - att.min()) * 255
    att= np.uint8(att)
    heatmap_img = cv2.applyColorMap(att, cv2.COLORMAP_INFERNO)
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(super_imposed_img.transpose((1,0,2)))
    ax[1].imshow(plot_pred(labels.detach().cpu()).squeeze().permute(1,0,2,3)[:,:,slice])
    for a in ax:
        a.axis('off')
    plt.show()


def get_contours(label):
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    contours = convolve2d(label, laplacian, mode="same")
    contours = 1. * (contours != 0)
    return contours

def grayscale_to_rgb(grayscale):
    return np.stack([grayscale, grayscale, grayscale], axis=-1)

def overlay_contours(image, contours):
    image[contours == 1.] = np.array([1., 0., 0.])
    return image


def plot_pred(pred):
    argmax = torch.argmax(pred, dim=1)
    color_img = torch.stack([argmax, argmax, argmax], dim=-1)
    color_img[argmax == 1] = torch.tensor([0,0,255])
    color_img[argmax == 2] = torch.tensor([0, 255, 255])
    color_img[argmax == 3] = torch.tensor([255, 0, 0])
    return color_img
####LOAD Baseline #####
baseline_dict = { "baseline_2018":  "baseline_2018_t1ce_fold1_quarter_0317_1114",
                     "baseline_2021" : "baseline_2021_t1ce_fold1_0317_1043",
                    "att_2018" : "student_bneck_2018_att_t1ce_fold1_0318_1144",
                    "kl_2018" : "student_bneck_2018_kl_t1ce_fold1_quarter_0318_1144",
                    "ct_2018" :  "student_2018_ct_t1ce_fold1_quarter_0318_1151",
                "kd_ct_2018" : "student_2018_ct_kd_t1ce_fold1_quarter_0318_1151",
                "kd_2018" : "student_bneck_2018_kd_t1ce_fold1_quarter_0318_1144"

}

student_dict = { "kl_kl_2018" : "student_bneck_2018_kl_kd_t1ce_fold1_quarter_0318_1144",
                 "kd_2021" : "student_bneck_2021_kd_t1ce_fold1_0318_1144",
                 "att_2021" :"student_bneck_2021_att_t1ce_fold1_0318_1144",
                 "kl_2021" : "student_bneck_2021_kl_t1ce_fold1_quarter_0318_1144",
                 "ct_2021" : "student_2021_ct_t1ce_fold1_0318_1151",
                "kd_ct_2021" : "student_2021_ct_kd_t1ce_fold1_0318_1151",
                "kl_kl_2021" : "student_bneck_2021_kl_kd_t1ce_fold1_0318_1144"
}
res_dir = os.path.join(os.environ["HOME"], "Nextcloud/knowledge-distillation/result/")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
for j, baseline_name in enumerate(baseline_dict.keys()):
    path_baseline = baseline_dict[baseline_name]
    checkpoint_name = 'Checkpoint_best_fold_1.pkl'
    checkpoint_path_baseline = os.path.join(res_dir, path_baseline, 'pkl', checkpoint_name)
    checkpoint_baseline = torch.load(checkpoint_path_baseline)
    config_baseline = checkpoint_baseline["config"]
    config_baseline['data_split_dir'] = os.path.join(os.environ["HOME"], "Nextcloud/knowledge-distillation/result/", path_baseline, "data_split")
    config_baseline['data_dir'] = os.path.join("/home/matthis/Nextcloud/knowledge-distillation/data")
    config_baseline['result_path'] = os.path.join(res_dir, path_baseline)
    config_baseline["device"] = device
    if baseline_name.split("_")[0] == "baseline":
        net_baseline = UNet(config_baseline).to(device)
    else:
        net_baseline = HUNetv4(config_baseline).to(device)
    net_baseline.load_state_dict(checkpoint_baseline["model_state_dict"])
    #### LOAD Student #####

    student_name = list(student_dict.keys())[j]
    path_student = student_dict[student_name]
    res_dir = os.path.join(os.environ["HOME"], "Nextcloud/knowledge-distillation/result/")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    checkpoint_name = 'Checkpoint_best_fold_1.pkl'
    checkpoint_path_student = os.path.join(res_dir, path_student, 'pkl', checkpoint_name)
    checkpoint_student = torch.load(checkpoint_path_student)
    config_student = checkpoint_student["config"]
    config_student['data_split_dir'] = os.path.join(os.environ["HOME"], "Nextcloud/knowledge-distillation/result/", path_student, "data_split")
    config_student['data_dir'] = os.path.join("/home/matthis/Nextcloud/knowledge-distillation/data")
    config_student['result_path'] = os.path.join(res_dir, path_student)
    config_student["device"] = device
    net_student = HUNetv4(config_student).to(device)
    net_student.load_state_dict(checkpoint_student["model_state_dict"])

    #### LOAD Data ####
    with open("../data/shared_test.txt", "r") as f:
        test_set = f.readlines()[0].split(",")
    config_baseline['data_dir'] = "/home/matthis/Nextcloud/knowledge-distillation/data/BraTS2021"
    dataset = TestBrats(config_baseline, test_set)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10)
    config_baseline['contrastive_loss'] = False
    criterion = TotalLoss(config_baseline)
    net_baseline.eval()
    progress_bar = tqdm(total=len(test_loader))

    with torch.no_grad():
        dice_et = []
        dice_tc = []
        dice_wt = []

        student_et = []
        student_tc = []
        student_wt = []

        size_et = []
        size_tc = []
        size_wt = []
        for i, data in enumerate(test_loader, 0):
            if i == 289: #or i==16 or i==120 or i==261:
            #if i>=0:
                imgs, labels = data
                inputs = [img.to(config_baseline['device']) for img in imgs]
                labels = labels.to(config_baseline['device'])
                baseline_pred = net_baseline(inputs, "target", temperature=False)
                student_pred, _, att_student, _ = net_student(inputs, "both", temperature=True)
                loss_dict, dice_baseline = criterion(baseline_pred, labels, similarity=False)
                loss_dict, dice_student = criterion(student_pred, labels, similarity=False)
                size_et.append(labels[0,3].sum().item())
                size_tc.append(size_et[-1] + labels[0, 1].sum().item())
                size_wt.append(size_tc[-1] + labels[0, 2].sum().item())
                print(dice_student)
                print(dice_baseline)
                #Sudent is the best: 2 = 221 and 1 = 319
                #Student is worse: 2=5 and 1=142
                if i == 10:
                    slice = 44
                elif i==13:
                    slice = 23
                elif i==16:
                    slice = 30
                elif i==261:
                    #cls = 1
                    slice = 21
                elif i==120:
                    #cls=2
                    slice = 31
                elif i == 289:
                    cls = 3
                elif i == 352:
                    cls = 1
                for slice in [56]:

                    #slice = torch.argmax((torch.argmax(student_pred[1], dim=1)[0] == cls).sum(dim=0).sum(dim=0))
                    #print(slice)
                    #if slice == 0:
                    #    slice = torch.argmax((torch.argmax(student_pred[1], dim=1)[0] == 2).sum(dim=0).sum(dim=0))

                    #fig, ax = plt.subplots(1,4)
                    image_to_show = inputs[1].squeeze().detach().cpu()
                    image_to_show = (image_to_show - image_to_show.min()) / (image_to_show.max() - image_to_show.min())


                    """plot_image = grayscale_to_rgb(image_to_show[:,:, slice])
                contours = get_contours(labels[0,cls,:, :, slice].detach().cpu())
                plot_image = overlay_contours(plot_image, contours)
                ax[0].imshow(plot_image.squeeze().transpose((1, 0, 2)))
                ax[0].axis('off')"""

                    """plot_image = grayscale_to_rgb(image_to_show[:,:, slice])
                contours = get_contours((baseline_pred[0].argmax(dim=1) == cls)[0, :,:, slice].detach().cpu() * 1.)
                plot_image = overlay_contours(plot_image, contours)
                ax[1].imshow(plot_image.squeeze().transpose((1, 0, 2)))
                ax[1].set_title('Dice ET: '+ str(dice_baseline['Dice_ET']))
                ax[1].axis('off')
    
                plot_image = grayscale_to_rgb(image_to_show[:,:, slice])
                contours = get_contours((student_pred[0].argmax(dim=1) == cls)[0, :,:, slice].detach().cpu() * 1.)
                plot_image = overlay_contours(plot_image, contours)
                ax[2].imshow(plot_image.squeeze().transpose((1, 0, 2)))
                ax[2].set_title('Dice ET: ' + str(dice_student['Dice_ET']))
                ax[2].axis('off')"""

                    plot_image = grayscale_to_rgb(image_to_show[:, :, slice])
                    """ax[0].imshow(plot_image.squeeze().transpose((1, 0, 2)))
                    ax[0].set_title("T1ce")
                    ax[0].axis('off')
                    ax[1].imshow(plot_pred(labels.detach().cpu()).squeeze().permute(1,0,2,3)[:,:,slice])
                    ax[1].set_title("Reference")
                    ax[1].axis('off')
                    ax[3].imshow(plot_pred(student_pred[1].detach().cpu()).squeeze().permute(1,0,2,3)[:, :, slice])
                    ax[3].set_title("Student")
                    ax[3].axis('off')
                    ax[2].imshow(plot_pred(baseline_pred[0].detach().cpu()).squeeze().permute(1,0,2,3)[:, :, slice])
                    ax[2].set_title("Baseline")
                    ax[2].axis('off')
                    plt.show()"""

                plt.imshow(plot_image.squeeze().transpose((1, 0, 2)))
                plt.axis('off')
                plt.savefig("/home/matthis/Images/brats/image_%d_2.png" % i, bbox_inches='tight', pad_inches=0)
                plt.close()
                plt.imshow(plot_pred(labels.detach().cpu()).squeeze().permute(1, 0, 2, 3)[:, :, slice])
                plt.axis('off')
                plt.savefig("/home/matthis/Images/brats/reference_%d_2.png" %i, bbox_inches='tight', pad_inches=0)
                plt.close()
                plt.imshow(plot_pred(baseline_pred[0].detach().cpu()).squeeze().permute(1,0,2,3)[:, :, slice])
                plt.axis('off')
                plt.savefig("/home/matthis/Images/brats/%s_%d_2.png" %(baseline_name, i), bbox_inches='tight', pad_inches=0)
                plt.close()
                plt.imshow(plot_pred(student_pred[1].detach().cpu()).squeeze().permute(1,0,2,3)[:, :, slice])
                plt.axis('off')
                plt.savefig("/home/matthis/Images/brats/%s_%d_2.png" %(student_name, i), bbox_inches='tight', pad_inches=0)
                plt.close()


                #plot_attention(image_to_show, att_student[1], slice, labels)

                #plot_attention(image_to_show, att_baseline[0], slice, labels)

                dice_et.append(dice_baseline['Dice_ET'])
                dice_tc.append(dice_baseline['Dice_TC'])
                dice_wt.append(dice_baseline['Dice_WT'])

                student_et.append(dice_student['Dice_ET'])
                student_tc.append(dice_student['Dice_TC'])
                student_wt.append(dice_student['Dice_WT'])
            progress_bar.update(1)
        dice_et = np.array(dice_et)
        dice_tc = np.array(dice_tc)
        dice_wt = np.array(dice_wt)

        student_et = np.array(student_et)
        student_tc = np.array(student_tc)
        student_wt = np.array(student_wt)
        print(dice_et.mean())
        print(dice_tc.mean())
        print(dice_wt.mean())

        print(student_et.mean())
        print(student_tc.mean())
        print(student_wt.mean())
        print((dice_et - student_et).max(), (dice_et - student_et).argmax())
        print((dice_wt - student_wt).max(), (dice_wt - student_wt).argmax())
        print((dice_et - student_et).min(), (dice_et - student_et).argmin())
        print((dice_wt - student_wt).min(), (dice_wt - student_wt).argmin())
        np.save("size_et.npy", size_et)
        np.save("size_tc.npy", size_tc)
        np.save("size_wt.npy", size_wt)
        progress_bar.close()




