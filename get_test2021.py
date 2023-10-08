import csv
import numpy as np
import pandas as pd

baseline_2021 = "../result/baseline_2021_t1ce_fold1_0317_1043"
baseline_2018 = "../result/baseline_2018_t1ce_fold1_0317_1038"

with open(baseline_2018 + "/data_split/train_set_fold_1.txt", "r") as f:
    train_list_2018 = f.readlines()[0].split(",")
    train_list_2018 = [image.split("/")[-1] for image in train_list_2018]

df = pd.read_csv("/home/matthis/Téléchargements/BraTS21-17_Mapping.csv")

path_mapping = pd.Series(df.BraTS2021.values, index=df.BraTS2018.values).to_dict()

for i, path in enumerate(train_list_2018):
    print(path, " becomes ", path_mapping[path])
    train_list_2018[i] = path_mapping[path]

with open(baseline_2021 + "/data_split/valid_set_fold_1.txt", "r") as f:
    test_list_2021 = f.readlines()[0].split(",")
    test_list_2021 = [image.split("/")[-1] for image in test_list_2021]
print(test_list_2021)

print("Number of element in BraTS 2021 test set:", len(test_list_2021))
print("Number of element in BraTS 2018 test set:", len(train_list_2018))

common_test = list(set(test_list_2021) - set(train_list_2018))
print("Number of shared element :", len(common_test))

with open("../data/shared_test.txt", "w") as f:
    f.writelines([",".join(common_test)])
