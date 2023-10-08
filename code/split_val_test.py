import os

import numpy as np
import csv
import glob
res_dir = "/home/matthis/Nextcloud/knowledge-distillation/result/"

def split_res(path):
    dice_et = np.load(os.path.join(path, "dice_et_post.npy"))
    dice_tc = np.load(os.path.join(path, "dice_tc.npy"))
    dice_wt = np.load(os.path.join(path, "dice_wt.npy"))
    pre_et = np.load(os.path.join(path, "pre_et_post.npy"))
    pre_tc = np.load(os.path.join(path, "pre_tc.npy"))
    pre_wt = np.load(os.path.join(path, "pre_wt.npy"))
    sens_et = np.load(os.path.join(path, "sens_et_post.npy"))
    sens_tc = np.load(os.path.join(path, "sens_tc.npy"))
    sens_wt = np.load(os.path.join(path, "sens_wt.npy"))
    spe_et = np.load(os.path.join(path, "spe_et_post.npy"))
    spe_tc = np.load(os.path.join(path, "spe_tc.npy"))
    spe_wt = np.load(os.path.join(path, "spe_wt.npy"))
    hauss_et = np.load(os.path.join(path, "hauss_et_post.npy"))
    hauss_tc = np.load(os.path.join(path, "hauss_tc.npy"))
    hauss_wt = np.load(os.path.join(path, "hauss_wt.npy"))
    l = len(dice_et)

    dice_et_valid, dice_et_test = dice_et[:l//2], dice_et[l//2:]
    dice_tc_valid, dice_tc_test = dice_tc[:l // 2], dice_tc[l // 2:]
    dice_wt_valid, dice_wt_test = dice_wt[:l // 2], dice_wt[l // 2:]

    pre_et_valid, pre_et_test = pre_et[:l // 2], pre_et[l // 2:]
    pre_tc_valid, pre_tc_test = pre_tc[:l // 2], pre_tc[l // 2:]
    pre_wt_valid, pre_wt_test = pre_wt[:l // 2], pre_wt[l // 2:]

    sens_et_valid, sens_et_test = sens_et[:l // 2], sens_et[l // 2:]
    sens_tc_valid, sens_tc_test = sens_tc[:l // 2], sens_tc[l // 2:]
    sens_wt_valid, sens_wt_test = sens_wt[:l // 2], sens_wt[l // 2:]

    spe_et_valid, spe_et_test = spe_et[:l // 2], spe_et[l // 2:]
    spe_tc_valid, spe_tc_test = spe_tc[:l // 2], spe_tc[l // 2:]
    spe_wt_valid, spe_wt_test = spe_wt[:l // 2], spe_wt[l // 2:]

    hauss_et_valid, hauss_et_test = hauss_et[:l // 2], hauss_et[l // 2:]
    hauss_tc_valid, hauss_tc_test = hauss_tc[:l // 2], hauss_tc[l // 2:]
    hauss_wt_valid, hauss_wt_test = hauss_wt[:l // 2], hauss_wt[l // 2:]

    return ([str(round(dice_et_valid.mean()*100, 2)) + " ("+ str(round(dice_et_valid.std() *100, 2)) + ")", str(round(dice_tc_valid.mean()*100, 2)) + " ("+ str(round(dice_tc_valid.std() *100, 2)) + ")", str(round(dice_wt_valid.mean()*100, 2)) + " ("+ str(round(dice_wt_valid.std() *100, 2)) + ")",
                                     str(round(pre_et_valid.mean()*100, 2)) + " ("+ str(round(pre_et_valid.std() *100, 2)) + ")", str(round(pre_tc_valid.mean()*100, 2)) + " ("+ str(round(pre_tc_valid.std() *100, 2)) + ")", str(round(pre_wt_valid.mean()*100, 2)) + " ("+ str(round(pre_wt_valid.std() *100, 2)) + ")",
                                     str(round(sens_et_valid.mean()*100, 2)) + " ("+ str(round(sens_et_valid.std() *100, 2)) + ")", str(round(sens_tc_valid.mean()*100, 2)) + " ("+ str(round(sens_tc_valid.std() *100, 2)) + ")", str(round(sens_wt_valid.mean()*100, 2)) + " ("+ str(round(sens_wt_valid.std() *100, 2)) + ")",
                                     str(round(spe_et_valid.mean()*100, 2)) + " ("+ str(round(spe_et_valid.std() *100, 2)) + ")", str(round(spe_tc_valid.mean()*100, 2)) + " ("+ str(round(spe_tc_valid.std() *100, 2)) + ")", str(round(spe_wt_valid.mean()*100, 2)) + " ("+ str(round(spe_wt_valid.std() *100, 2)) + ")",
                                     str(round(hauss_et_valid.mean(), 2)) + " ("+ str(round(hauss_et_valid.std(), 2)) + ")", str(round(hauss_tc_valid.mean(), 2)) + " ("+ str(round(hauss_tc_valid.std(), 2)) + ")", str(round(hauss_wt_valid.mean(), 2)) + " ("+ str(round(hauss_wt_valid.std(), 2)) + ")"],
                                    [str(round(dice_et_test.mean()*100, 2)) + " ("+ str(round(dice_et_test.std() *100, 2)) + ")", str(round(dice_tc_test.mean()*100, 2)) + " ("+ str(round(dice_tc_test.std() *100, 2)) + ")", str(round(dice_wt_test.mean()*100, 2)) + " ("+ str(round(dice_wt_test.std() *100, 2)) + ")",
                                     str(round(pre_et_test.mean()*100, 2)) + " ("+ str(round(pre_et_test.std() *100, 2)) + ")", str(round(pre_tc_test.mean()*100, 2)) + " ("+ str(round(pre_tc_test.std() *100, 2)) + ")", str(round(pre_wt_test.mean()*100, 2)) + " ("+ str(round(pre_wt_test.std() *100, 2)) + ")",
                                     str(round(sens_et_test.mean()*100, 2)) + " ("+ str(round(sens_et_test.std() *100, 2)) + ")", str(round(sens_tc_test.mean()*100, 2)) + " ("+ str(round(sens_tc_test.std() *100, 2)) + ")", str(round(sens_wt_test.mean()*100, 2)) + " ("+ str(round(sens_wt_test.std() *100, 2)) + ")",
                                     str(round(spe_et_test.mean()*100, 2)) + " ("+ str(round(spe_et_test.std() *100, 2)) + ")", str(round(spe_tc_test.mean()*100, 2)) + " ("+ str(round(spe_tc_test.std() *100, 2)) + ")", str(round(spe_wt_test.mean()*100, 2)) + " ("+ str(round(spe_wt_test.std() *100, 2)) + ")",
                                     str(round(hauss_et_test.mean(), 2)) + " ("+ str(round(hauss_et_test.std(), 2)) + ")", str(round(hauss_tc_test.mean(), 2)) + " ("+ str(round(hauss_tc_test.std(), 2)) + ")", str(round(hauss_wt_test.mean(), 2)) + " ("+ str(round(hauss_wt_test.std(), 2)) + ")"])

models = {"teacher":[""], "baseline":[""], "student":["att", "kd", "ct", "ct_kd"], "student_bneck":["att", "kl", "kl_kd", "kd"]}
years = ["_2018", "_2021"]
datas = ["_0", "_half", "_quarter"]
losses = [ "att", "kl", "kl_kd", "kd", "ct", "ct_kd"]
dir= "/home/matthis/Nextcloud/knowledge-distillation/result/"

for year in years:
    for data in datas:
        for model in models.keys():
            if model.split("_")[0] in ["student"]:
                for loss in models[model]:
                    print(dir + model + year + "_" + loss + "_t1ce" +"_fold1" + data + "*")
                    file = glob.glob(dir + model + year + "_" + loss + "_t1ce" +"_fold1" + data + "*")
                    if len(file) > 0:
                        file = file[0]
                        test_res, valid_res = split_res(file)
                        with open(os.path.join(os.environ["HOME"], "Nextcloud/knowledge-distillation", "save_test.csv"), "a") as save_file:
                            writer = csv.writer(save_file)
                            writer.writerow([year + " " + data + " " + loss] + test_res)
            elif model == "teacher":
                if data == "_0":
                    file = dir + model + year + "_t1ce" +"_fold1"
                else:
                    file = dir + model + year  + "_t1ce" + "_fold1" + data
                test_res, valid_res = split_res(file)
                with open(os.path.join(os.environ["HOME"], "Nextcloud/knowledge-distillation", "save_test.csv"),
                          "a") as save_file:
                    writer = csv.writer(save_file)
                    writer.writerow([model + " " + year + " " + data] + test_res)
            else:
                file = glob.glob(dir + model + year + "_t1ce_fold1" + data + "*")
                if len(file) > 0:
                    file = file[0]
                    test_res, valid_res = split_res(file)
                    with open(os.path.join(os.environ["HOME"], "Nextcloud/knowledge-distillation", "save_test.csv"),
                              "a") as save_file:
                        writer = csv.writer(save_file)
                        writer.writerow([model + " " + year + " " + data] + test_res)