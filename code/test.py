import os
import torch
import numpy as np
from tqdm import tqdm
import csv

from models.HUnet import UNet, HUNetv4
from dataset_brats import TestBrats
from loss_func import TotalLoss, TestLoss

res_dir = os.path.join(os.environ["HOME"], "Nextcloud/knowledge-distillation/result/")
device = "cuda" if torch.cuda.is_available() else "cpu"
count = 0
for i, path in enumerate(os.listdir(res_dir)):
    if len(path.split("_")) > 1:
        #if path in ["student_2018_kl_kd_t1ce_fold1_0318_1121", "baseline_2018_t1ce_fold1_0317_1038"]:
        print(path)
        fold = 1
        checkpoint_name = 'Checkpoint_best_fold_%d.pkl' % fold
        checkpoint_path = os.path.join(res_dir, path, 'pkl', checkpoint_name)
        checkpoint = torch.load(checkpoint_path)
        config = checkpoint["config"]
        res_path = config['result_path'].split('/')
        for j in range(len(res_path)):
            if res_path[j] == "results":
                res_path[j] ="result"
        config['result_path'] = os.path.join(os.environ["HOME"], "Nextcloud/knowledge-distillation/result", path)
        config["device"] = device
        if path.split("_")[0] == "baseline" or path.split("_")[0] == "teacher":
            net = UNet(config).to(device)
        else:
            net = HUNetv4(config).to(device)
        config['data_dir'] = os.path.join(os.environ["HOME"], "Nextcloud", "/".join(config['data_dir'].split("/")[5:]))
        net.load_state_dict(checkpoint["model_state_dict"])
        """if path.split("_")[-3] == "half":
            valid_set_file = os.path.join(os.environ["HOME"], "knowledge-distillation/test_set_half.txt")
        if path.split("_")[-3] == "quarter":
            valid_set_file = os.path.join(os.environ["HOME"], "knowledge-distillation/test_set_quarter.txt")"""
        with open("../data/shared_test.txt", "r") as f:
            test_set = f.readlines()[0].split(",")
        config['data_dir'] = "/home/matthis/Nextcloud/knowledge-distillation/data/BraTS2021"
        dataset = TestBrats(config, test_set)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10)
        criterion = TestLoss(config)
        net.eval()
        progress_bar = tqdm(total=len(test_loader))
        with torch.no_grad():
            dice_et = []
            dice_tc = []
            dice_wt = []
            pre_et = []
            pre_tc = []
            pre_wt = []
            sens_et = []
            sens_tc = []
            sens_wt = []
            spe_et = []
            spe_tc = []
            spe_wt = []
            hauss_et = []
            hauss_tc = []
            hauss_wt = []
            for i, data in enumerate(test_loader, 0):
                if False:
                    if i >= 5:
                        break
                imgs, labels = data
                inputs = [img.to(config['device']) for img in imgs]
                labels = labels.to(config['device'])
                if path.split("_")[0] == "teacher":
                    y_pred = net(inputs, "source")
                else:
                    y_pred = net(inputs, "target")
                loss_dict, dice_dict, pre, sens, spe, hauss = criterion(y_pred, labels, similarity=False)
                dice_et.append(dice_dict['Dice_ET'])
                dice_tc.append(dice_dict['Dice_TC'])
                dice_wt.append(dice_dict['Dice_WT'])

                pre_et.append(pre['ET'])
                pre_tc.append(pre['TC'])
                pre_wt.append(pre['WT'])
                sens_et.append(sens['ET'])
                sens_tc.append(sens['TC'])
                sens_wt.append(sens['WT'])
                spe_et.append(spe['ET'])
                spe_tc.append(spe['TC'])
                spe_wt.append(spe['WT'])
                hauss_et.append(hauss['ET'])
                hauss_tc.append(hauss['TC'])
                hauss_wt.append(hauss['WT'])

                progress_bar.update(1)
            dice_et = np.array(dice_et)
            dice_tc = np.array(dice_tc)
            dice_wt = np.array(dice_wt)
            print(dice_et.mean())
            print(dice_tc.mean())
            print(dice_wt.mean())
            pre_et = np.array(pre_et)
            pre_tc = np.array(pre_tc)
            pre_wt = np.array(pre_wt)
            sens_et = np.array(sens_et)
            sens_tc = np.array(sens_tc)
            sens_wt = np.array(sens_wt)
            spe_et = np.array(spe_et)
            spe_tc = np.array(spe_tc)
            spe_wt = np.array(spe_wt)
            hauss_et = np.array(hauss_et)
            hauss_tc = np.array(hauss_tc)
            hauss_wt = np.array(hauss_wt)
            np.save(os.path.join(config['result_path'], "dice_et.npy"), dice_et)
            np.save(os.path.join(config['result_path'], "dice_tc.npy"), dice_tc)
            np.save(os.path.join(config['result_path'], "dice_wt.npy"), dice_wt)
            np.save(os.path.join(config['result_path'], "pre_et.npy"), pre_et)
            np.save(os.path.join(config['result_path'], "pre_tc.npy"), pre_tc)
            np.save(os.path.join(config['result_path'], "pre_wt.npy"), pre_wt)
            np.save(os.path.join(config['result_path'], "sens_et.npy"), sens_et)
            np.save(os.path.join(config['result_path'], "sens_tc.npy"), sens_tc)
            np.save(os.path.join(config['result_path'], "sens_wt.npy"), sens_wt)
            np.save(os.path.join(config['result_path'], "spe_et.npy"), spe_et)
            np.save(os.path.join(config['result_path'], "spe_tc.npy"), spe_tc)
            np.save(os.path.join(config['result_path'], "spe_wt.npy"), spe_wt)
            np.save(os.path.join(config['result_path'], "hauss_et.npy"), hauss_et)
            np.save(os.path.join(config['result_path'], "hauss_tc.npy"), hauss_tc)
            np.save(os.path.join(config['result_path'], "hauss_wt.npy"), hauss_wt)

            with open(os.path.join(os.environ["HOME"], "Nextcloud/knowledge-distillation", "common_test.csv"), "a") as save_file:
                writer = csv.writer(save_file)

                writer.writerow([path, str(round(dice_et.mean()*100, 2)) + " ("+ str(round(dice_et.std() *100, 2)) + ")", str(round(dice_tc.mean()*100, 2)) + " ("+ str(round(dice_tc.std() *100, 2)) + ")", str(round(dice_wt.mean()*100, 2)) + " ("+ str(round(dice_wt.std() *100, 2)) + ")",
                                 str(round(pre_et.mean()*100, 2)) + " ("+ str(round(pre_et.std() *100, 2)) + ")", str(round(pre_tc.mean()*100, 2)) + " ("+ str(round(pre_tc.std() *100, 2)) + ")", str(round(pre_wt.mean()*100, 2)) + " ("+ str(round(pre_wt.std() *100, 2)) + ")",
                                 str(round(sens_et.mean()*100, 2)) + " ("+ str(round(sens_et.std() *100, 2)) + ")", str(round(sens_tc.mean()*100, 2)) + " ("+ str(round(sens_tc.std() *100, 2)) + ")", str(round(sens_wt.mean()*100, 2)) + " ("+ str(round(sens_wt.std() *100, 2)) + ")",
                                 str(round(spe_et.mean()*100, 2)) + " ("+ str(round(spe_et.std() *100, 2)) + ")", str(round(spe_tc.mean()*100, 2)) + " ("+ str(round(spe_tc.std() *100, 2)) + ")", str(round(spe_wt.mean()*100, 2)) + " ("+ str(round(spe_wt.std() *100, 2)) + ")",
                                 str(round(hauss_et.mean(), 2)) + " ("+ str(round(hauss_et.std(), 2)) + ")", str(round(hauss_tc.mean(), 2)) + " ("+ str(round(hauss_tc.std(), 2)) + ")", str(round(hauss_wt.mean(), 2)) + " ("+ str(round(hauss_wt.std(), 2)) + ")"])
            #save_file.write(path + " - ET : %f ± %f, TC : %f ± %f, WT : %f ± %f \n"%(dice_et.mean(), dice_et.std(), dice_tc.mean(), dice_tc.std(), dice_wt.mean(), dice_wt.std()))
            progress_bar.close()
