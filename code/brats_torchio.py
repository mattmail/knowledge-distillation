import os
import random
import torch
import torchio as tio
import shutil
import numpy as np
from tools.log import add_log
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, KFold

class RandomCrop:
    def __init__(self, shape=(128,128,128)):
        self.shape = shape

    def __call__(self, sample):
        sampler = tio.data.UniformSampler(self.shape)
        patch = list(sampler(sample, 1))[0]
        return patch

def prepare_data(config, shuffle=True):
    print('==> Preparing data...')
    if config['data_split'] == 'cross_validation':
        n_fold_split(config)
        #copy_data_split(config, os.path.join(os.environ['HOME'] + "/knowledge-distillation/", 'results',
        #                                    'teacher_t1ce_fold1_0226_1501'))
    elif config['data_split'] == 'default':
        copy_data_split(config, '../default')
    elif config['data_split'] == 'same_as_teacher':
        copy_data_split(config, os.path.join(os.environ['HOME'] + "/Nextcloud/knowledge-distillation/", 'result',
                                             config['pretrained_teacher']))

    config['data_dir'] = os.path.join(config['workspace_root'], 'data')
    if config['dataset'] == 'BraTS_2018':
        config['data_dir'] = os.path.join(config['data_dir'], 'BraTS_2018')
    elif config['dataset'] == 'BraTS_2019':
        config['data_dir'] = os.path.join(config['data_dir'], 'MICCAI_BraTS_2019_Data_Training')
    elif config['dataset'] == 'BraTS_2020':
        config['data_dir'] = os.path.join(config['data_dir'], 'MICCAI_BraTS_2020_Data_Training')
    elif config['dataset'] == 'BraTS2021':
        config['data_dir'] = os.path.join(config['data_dir'], 'BraTS2021')
    else:
        raise Exception("Invalid data set name ! ")

    train_set = BratsDataset(config, 'train')
    valid_set = BratsDataset(config, 'valid')

    add_log(config, 'Fold No.%d\t Total data: %d\t train data: %d\t validation data: %d\t' %
            (config['fold'], len(train_set) + len(valid_set), len(train_set), len(valid_set)))

    num_workers = 1 if config['debug_mode'] else 10
    num_workers = 0
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=shuffle,
                                               num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=config['batch_size'], shuffle=False,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(valid_set, batch_size=config['batch_size'], shuffle=False,
                                              num_workers=num_workers)

    return train_loader, valid_loader, test_loader

def BratsDataset(config, mode):

    data_dir = config['data_dir']
    dataset_file = os.path.join(config['data_split_dir'],
                                "{mode}_set_fold_{fold}.txt".format(mode=mode, fold=config['fold']))
    with open(dataset_file, 'r') as f:
        subject_list = f.readlines()[0].split(",")
    if config['use_half']:
        subject_list = subject_list[:int(len(subject_list) / 2)]
    if config['use_quarter']:
        subject_list = subject_list[:int(len(subject_list) / 4)]
    images = []
    for subject in subject_list:
        path = subject + '/' +  subject.split('/')[-1]
        image = tio.Subject(
            t1 =tio.ScalarImage(path + "_t1.nii.gz"),
        t1ce =tio.ScalarImage(path + "_t1ce.nii.gz"),
        t2 =tio.ScalarImage(path + "_t2.nii.gz"),
        flair =tio.ScalarImage(path + "_flair.nii.gz"),
        seg = tio.LabelMap(path + "_seg.nii.gz")
        )
        images.append(image)

    if mode == "train":
        transforms = tio.Compose([
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            tio.CropOrPad((192, 192, 144)),
            tio.RandomNoise(p=0.5),
            tio.RandomFlip(),
            tio.OneOf({
                tio.RandomAffine(): 0.2,
                tio.RandomElasticDeformation(): 0.2,
            })
            #RandomCrop(config['img_size'])
        ])
        dataset = tio.SubjectsDataset(images, transform=transforms)
        sampler = tio.data.UniformSampler(config['img_size'])
        return tio.data.Queue(dataset, max_length=100, samples_per_volume=10, sampler=sampler, num_workers=10)
    if mode == "valid" or mode == "test":
        transforms = tio.Compose([
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            tio.CropOrPad((192, 192, 144))
        ])
        dataset = tio.SubjectsDataset(images, transform=transforms)
        return dataset




def seg_label(seg, softmax=False):
    """To transform the segmentation to label.

    Args:
        seg: annotations which use different values to segment the data. (0-background, 1-non-enhancing tumor core(NET),
        2-the peritumoral edema(ED), 4-the GD-enhancing tumor(ET)).

        ET = ET
        TC = ET + NET/NCR
        WT = ET + NET/NCR + ED

    Returns:
        A numpy array contains 3 channels(ET, TC, WT). In each channel, pixels are labeled as 0 or 1.
    """

    if softmax:
        label_0 = 1 * (seg == 0)
        label_1 = 1 * (seg == 1)
        label_2 = 1 * (seg == 2)
        label_4 = 1 * (seg == 4)
        labels = [label_0, label_1, label_2, label_4]
    else:
        label_ET = np.zeros(seg.shape)
        label_ET[np.where(seg == 4)] = 1
        label_TC = np.zeros(seg.shape)
        label_TC[np.where((seg == 1) | (seg == 4))] = 1
        label_WT = np.zeros(seg.shape)
        label_WT[np.where(seg > 0)] = 1

        labels = [label_ET, label_TC, label_WT]
    return np.asarray(labels, dtype=np.float32)





def n_fold_split(config):
    """ Split data set into K folds and save it in text files"""

    config['data_dir'] = os.path.join(config['workspace_root'], 'data')
    config['data_dir'] = os.path.join(config['data_dir'], config['dataset'])

    if config['dataset'] == 'BraTS_2018':

        HGG_dir = os.path.join(config['data_dir'], 'HGG')
        LGG_dir = os.path.join(config['data_dir'], 'LGG')

        HGG_patient_18 = os.listdir(HGG_dir)
        LGG_patient_18 = os.listdir(LGG_dir)

        data_set_HGG = [os.path.join(HGG_dir, patient) for patient in HGG_patient_18]
        data_set_LGG = [os.path.join(LGG_dir, patient) for patient in LGG_patient_18]

        # shuffle data set
        r = random.random
        random.seed(41)
        random.shuffle(data_set_HGG, random=r)
        random.shuffle(data_set_LGG, random=r)

        # build K fold
        n = config['num_fold']
        n_fold = [data_set_HGG[i::n] + data_set_LGG[i::n] for i in range(n)]

        if not os.path.exists(config['data_split_dir']):
            os.mkdir(config['data_split_dir'])

        for i in range(n):
            train_set_path = os.path.join(config['data_split_dir'], 'train_set_fold_%d.txt' % (i + 1))
            valid_set_path = os.path.join(config['data_split_dir'], 'valid_set_fold_%d.txt' % (i + 1))

            valid_set = n_fold[i]
            train_set = list(set(data_set_HGG + data_set_LGG) - set(valid_set))
            # save splitting
            with open(train_set_path, 'w') as f:
                f.writelines(",".join(train_set))
            with open(valid_set_path, 'w') as f:
                f.writelines(",".join(valid_set))

    elif config['dataset'] in ['BraTS_2020', 'BraTS_2019']:
        name_mapping = os.path.join(config['data_dir'], 'name_mapping.csv')
        # select only Brats_id and grade
        mapping = np.loadtxt(name_mapping, delimiter=",", dtype='U', usecols=(0, 5))
        # remove first line that contains column names
        mapping = mapping[1:, ...]

        # all subject
        # data_set = [os.path.join(config['data_dir'], subject) for subject in mapping [:,1]]
        data_set = [subject for subject in mapping[:, 1]]
        grade_set = [grade for grade in mapping[:, 0]]

        # Init the Kfold
        skf = StratifiedKFold(n_splits=config['num_fold'], shuffle=True)

        if not os.path.exists(config['data_split_dir']):
            os.mkdir(config['data_split_dir'])

        for i, (train_index, test_index) in enumerate(skf.split(data_set, grade_set)):
            # create path to train/test fold txt files
            train_set_path = os.path.join(config['data_split_dir'], 'train_set_fold_%d.txt' % (i + 1))
            valid_set_path = os.path.join(config['data_split_dir'], 'valid_set_fold_%d.txt' % (i + 1))

            train_set = np.array(data_set)[train_index]
            valid_set = np.array(data_set)[test_index]

            # save train set and test set in a text file
            with open(train_set_path, 'w') as f:
                f.writelines(",".join(train_set))
            with open(valid_set_path, 'w') as f:
                f.writelines(",".join(valid_set))
    elif config['dataset'] == 'BraTS2021':

        data_set = [subject for subject in os.listdir(config['data_dir']) if
                    subject[:5] == "BraTS" or subject[:5] == "Brats"]

        # Init the Kfold
        skf = KFold(n_splits=config['num_fold'], shuffle=True)

        if not os.path.exists(config['data_split_dir']):
            os.mkdir(config['data_split_dir'])

        for i, (train_index, test_index) in enumerate(skf.split(data_set)):
            # create path to train/test fold txt files
            train_set_path = os.path.join(config['data_split_dir'], 'train_set_fold_%d.txt' % (i + 1))
            valid_set_path = os.path.join(config['data_split_dir'], 'valid_set_fold_%d.txt' % (i + 1))

            train_set = np.array(data_set)[train_index]
            valid_set = np.array(data_set)[test_index]

            # save train set and test set in a text file
            with open(train_set_path, 'w') as f:
                f.writelines(",".join(train_set))
            with open(valid_set_path, 'w') as f:
                f.writelines(",".join(valid_set))


def copy_data_split(config, file_dir):
    source = os.path.join(file_dir, "data_split")
    target = os.path.join(config['result_path'], "data_split")
    if not os.path.exists(target):
        shutil.copytree(source, target)
        add_log(config, "Copy data split from %s." % file_dir)



































