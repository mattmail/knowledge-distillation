
import os 
import random
import torch
import shutil
import numpy as np 
import nibabel as nib
from tools.log import add_log
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from configuration import configuration
from scipy import interpolate
import cv2

def prepare_data(config, shuffle=True):
    print('==> Preparing data...')
    """if config['data_split'] == 'cross_validation':
        #n_fold_split(config)
        copy_data_split(config, os.path.join(os.environ['HOME'] + "/Nextcloud/knowledge-distillation/", 'result', 'baseline_t1ce_fold1_1220_1506'))
    elif config['data_split'] == 'default':
        copy_data_split(config, '../default')
    elif config['data_split'] == 'same_as_teacher':
        copy_data_split(config, os.path.join(os.environ['HOME'] + "/Nextcloud/knowledge-distillation/", 'result', config['pretrained_teacher']))
"""
    #config['data_dir'] = os.path.join(config['workspace_root'], 'data')
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

    train_set = BratsDataset(config, mode='train')
    valid_set = BratsDataset(config, mode='valid')
    
    
    add_log(config, 'Fold No.%d\t Total data: %d\t train data: %d\t validation data: %d\t' %
            (config['fold'], len(train_set) + len(valid_set), len(train_set), len(valid_set)))

    num_workers = 1 if config['debug_mode'] else 10
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=shuffle, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=config['batch_size'], shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(valid_set, batch_size=config['batch_size'], shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader
    
class BratsDataset(Dataset):
    def __init__(self, config, mode):

        super(BratsDataset, self).__init__()
        self.config = config
        self.subject_list = []
        self.loaded_subjects ={} 
        self.data_dir = None
        self.data_2d_dir = None 
        self.data_2d_img_dir = None
        self.data_2d_seg_dir = None
        self.mode = mode
            
        
        if self.config["img_dim"] == "3D":
            self.data_dir = config['data_dir']
            #dataset_file = os.path.join(self.config['data_split_dir'], "{mode}_set_fold_{fold}.txt".format(mode=mode, fold=config['fold']))
            #with open(dataset_file, 'r') as f:
            #    self.subject_list = f.readlines()[0].split(",")
            #if config['use_half']:
            #    #random.shuffle(self.subject_list)
            #    self.subject_list = self.subject_list[:int(len(self.subject_list)/2)]
            #if config['use_quarter']:
            #    #random.shuffle(self.subject_list)
            #    self.subject_list = self.subject_list[:int(len(self.subject_list)/4)]
        
        else : 
            raise Exception ("Img_dim value is invalid!")

    def load_data(self, index, return_filename=False):
        """To load image data and segmentation annotaions"""

        if self.config["img_dim"] == "2D":
            subject = self.subject_list[index][0]
            index_slice = self.subject_list[index][1] 
            img = np.load(os.path.join(self.data_2d_img_dir, subject+'_img_slice_{}.npy'.format(index_slice) ))
            seg = np.load(os.path.join(self.data_2d_seg_dir, subject+'_seg_slice_{}.npy'.format(index_slice) ))
            if self.config['half_size']:
                img = img[:, ::2, ::2]
                seg = seg[::2, ::2]

        else : #3D 
            subject = os.path.join(self.data_dir, self.subject_list[index])
            nii_list = []
            for mode in self.config['modality_list']:
                nii_list.append(load_nii(subject, mode))
            img = np.array(nii_list, dtype=np.float32)
            seg = np.array(load_nii(subject, 'seg'))
            # make sure we have the right image size
        if return_filename:
            return img, seg, subject
        else:
            return img, seg

    def __getitem__(self, index):
        if self.config['return_filename']:
            img, seg, filename = self.load_data(index, self.config['return_filename'])
        else:
            img, seg = self.load_data(index)
            #img, seg = self.image_list[index]

                
        # pre-processing & augmentation
        if self.config['random_mirror_flip'] and self.mode == "train":
            img, seg = mirror_flip(img, seg, config=self.config)
        if self.config['data_augmentation'] and self.mode == "train":
            if random.random() > 0.3:
                img = augment_gamma(img)
        #if self.config['data_preprocessing']:
        #    img = scale_non_zero(img, config=self.config)
        img, seg = crop(img,seg, shape=(192,192,144)) #roughly eliminate zero-elements in the image
        #if self.config['crop_the_void_3D_img'] and self.mode == "train":
        #    img, seg = crop(img,seg,config=self.config)
        label = seg_label(seg, self.config['softmax'])

        if self.config['half_size']:
            img = img[:, ::2, ::2, ::2]
            label = label[:, ::2, ::2, ::2]

        img_T1 = torch.from_numpy(img[0][np.newaxis, ...].astype(np.float32)).type('torch.FloatTensor')
        img_T1ce = torch.from_numpy(img[1][np.newaxis, ...].astype(np.float32)).type('torch.FloatTensor')
        img_T2 = torch.from_numpy(img[2][np.newaxis, ...].astype(np.float32)).type('torch.FloatTensor')
        img_Flair = torch.from_numpy(img[3][np.newaxis, ...].astype(np.float32)).type('torch.FloatTensor')
        label = torch.from_numpy(label)
        if self.config['return_filename']:
            return [img_T1, img_T1ce, img_T2, img_Flair], label, filename
        return [img_T1, img_T1ce, img_T2, img_Flair], label

    def __len__(self):
        return len(self.subject_list)
def crop(img, seg, config=None, shape=None):
    """
    Cropping a 3D image 
    """
    if config is not None:
        if config['crop_mode'] == "center":
            c1_l = int((img.shape[1] - config['img_size'][0])/2)
            c2_l = int((img.shape[2] - config['img_size'][1])/2)
            c3_l = int((img.shape[3] - config['img_size'][2])/2)
        elif config['crop_mode'] == "random":
            c1_l = random.randint(0, img.shape[1] - config['img_size'][0])
            c2_l = random.randint(0, img.shape[2] - config['img_size'][1])
            c3_l = random.randint(0, img.shape[3] - config['img_size'][2])
        c1_r = int(c1_l + config['img_size'][0])
        c2_r = int(c2_l + config['img_size'][1])
        c3_r = int(c3_l + config['img_size'][2])

    elif shape is not None:
        c1_l = int((img.shape[1] - shape[0])/2)
        c2_l = int((img.shape[2] - shape[1])/2)
        c3_l = int((img.shape[3] - shape[2])/2)

        c1_r = int(c1_l + shape[0])
        c2_r = int(c2_l + shape[1])
        c3_r = int(c3_l + shape[2])

    
    img = img[:, c1_l: c1_r, c2_l: c2_r, c3_l:c3_r]
    seg = seg[c1_l: c1_r, c2_l: c2_r, c3_l:c3_r]

    #img = np.pad(img[:, c1_l: c1_r, c2_l: c2_r], ((0,0),(0,0),(0,0),(2,2)))
    #seg = np.pad(seg[c1_l: c1_r, c2_l: c2_r], ((0,0),(0,0),(2,2)))

    return img, seg

def cropping_the_void_in_3D_img(img,seg,config):
    """
    Cropping the 3D image based on the given new image size .  

    """

    if len(config["void_crop_bbox"]) == 6:
        Xmax,Xmin, Ymax,Ymin,Zmax,Zmin = config["void_crop_bbox"]
        img = img[: ,Xmin: Xmax+1, Ymin:Ymax+1, Zmin:Zmax+1   ]
        seg = seg[Xmin: Xmax+1, Ymin:Ymax+1, Zmin:Zmax+1 ]
        new_img = np.zeros(tuple([img.shape[0]]+config["3D_img_size_after_cropping"]))
        new_seg = np.zeros(tuple(config["3D_img_size_after_cropping"]))
        x1,X2 = 0, min(img.shape[1],new_img.shape[1]) 
        y1,y2 = 0, min(img.shape[2],new_img.shape[2]) 
        z1,z2 = 0, min(img.shape[3],new_img.shape[3])
        new_img[:,x1:X2,y1:y2,z1:z2]  = img[:,x1:X2,y1:y2,z1:z2]
        new_seg[x1:X2,y1:y2,z1:z2]  = seg[x1:X2,y1:y2,z1:z2]
        
        return new_img ,new_seg 
    else:
        raise Exception ("Invalid lenght!" )

def read_indices_2D_images_file(dataset_file):

    subject_list = [] 
    with open(dataset_file, 'r') as f:
        lines = f.readlines()
        for line in lines: 
            line= line.split(",")
            subject =  line.pop(0)
            # remove "\n"
            line.pop(-1)
            subject_list += [(subject,int(element)) for element in line]
    return(subject_list)
            

def intensity_shift_non_zero(img):
    """A random (per channel) intensity shift (-0.1 .. 0.1 of image std) on input images.

    Args:
        img: multi-channel brats image data.

    Returns:
        image after random intensity shift, same size as input.
    """
    for i, channel in enumerate(img):
        shift = random.random() * 0.2 - 0.1  # random intensity shift (-0.1..0.1 of image std)
        count = np.zeros(channel.shape)
        count[np.where(channel > 0)] = 1
        img[i] = channel + shift * count
    return img

def augment_gamma(img, gamma_range=(0.7, 1.5)):
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    img_min = img.min()
    range = img.max()-img_min
    normalized = (img-img_min/range)
    gamma_img = np.sign(normalized) * np.abs(normalized)**gamma * range + img_min
    return gamma_img


def scale_non_zero(imgs, config, eps = 10**(-30)):
    """Scale each channel to z-score on non-zero voxels only.

    Args:
        img: multi-channel brats image data.

    Returns:
        image after scaling, same size as input.
    """

    for i, img in enumerate(imgs):
        mask = img != 0
        roi_arr = img[mask]
        roi_arr = roi_arr[roi_arr < config['upper_clip']]
        mean = roi_arr.mean()
        std = roi_arr.std()
        img = img.astype(dtype=np.float32)
        img = (img - mean) / (std + 1e-10)
        imgs[i] = img
    return imgs

def mirror_flip(img, label, config):
    """A random axis mirror flip (for all 3 axes) with a probability 0.5.

    Args:
        img: multi-channel brats image data.

    Returns:
        image after random flips, same size as input.
    """
    if random.random() < 0.3:
        if config["img_dim"] == "3D": 
            img = img[:, ::-1, :, :].copy()
            label = label[:, ::-1, :, :].copy()
        else: # 2D 
            img = img[:, ::-1, :].copy()
            label = label[:, ::-1, :].copy()   
    if random.random() < 0.3:
        if config["img_dim"] == "3D": 
            img = img[:, :, ::-1, :].copy()
            label = label[:, :, ::-1, :].copy()
        else : #2D 
            img = img[:, :, ::-1].copy()
            label = label[:, :, ::-1].copy()   
    if config["img_dim"] == "3D":        
        if random.random() < 0.5:
            img = img[:, :, :, ::-1].copy()
            label = label[:, :, :, ::-1].copy()
    return img, label



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
def load_nii(subject, mode):
    """To load NIFTI file"""
    subject_path = subject
    subject_name = os.listdir(subject_path)[0].split('.')[0].split('_')[:-1]
    subject_name.append(mode)
    img_full_name = '_'.join(subject_name) + '.nii.gz'
    img_path = os.path.join(subject_path, img_full_name)
    return nib.load(img_path).get_fdata()

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
            train_set_path = os.path.join(config['data_split_dir'], 'train_set_fold_%d.txt' % (i+1))
            valid_set_path = os.path.join(config['data_split_dir'], 'valid_set_fold_%d.txt' % (i+1))

            valid_set = n_fold[i]
            train_set = list(set(data_set_HGG + data_set_LGG) - set(valid_set))
            # save splitting 
            with open(train_set_path, 'w') as f:
                f.writelines(",".join(train_set))
            with open(valid_set_path, 'w') as f:
                f.writelines(",".join(valid_set))

    elif config['dataset'] in ['BraTS_2020',  'BraTS_2019']:
        name_mapping = os.path.join(config['data_dir'], 'name_mapping.csv')
        # select only Brats_id and grade 
        mapping = np.loadtxt(name_mapping, delimiter=",", dtype='U', usecols=(0,5))
        # remove first line that contains column names 
        mapping = mapping[1:,...]
        
        # all subject 
        #data_set = [os.path.join(config['data_dir'], subject) for subject in mapping [:,1]] 
        data_set = [subject for subject in mapping [:,1]] 
        grade_set = [grade for grade in mapping[:,0]]

        # Init the Kfold 
        skf = StratifiedKFold(n_splits=config['num_fold'], shuffle=True)

        if not os.path.exists(config['data_split_dir']):
            os.mkdir(config['data_split_dir'])

        for i, (train_index, test_index) in enumerate(skf.split(data_set, grade_set)):
            # create path to train/test fold txt files 
            train_set_path = os.path.join(config['data_split_dir'], 'train_set_fold_%d.txt' % (i+1))
            valid_set_path = os.path.join(config['data_split_dir'], 'valid_set_fold_%d.txt' % (i+1))

            train_set =  np.array(data_set)[train_index]
            valid_set = np.array(data_set)[test_index]

            # save train set and test set in a text file    
            with open(train_set_path, 'w') as f:
                f.writelines(",".join(train_set))
            with open(valid_set_path, 'w') as f:
                f.writelines(",".join(valid_set))
    elif config['dataset'] == 'BraTS2021':

        data_set = [subject for subject in os.listdir(config['data_dir']) if subject[:5] == "BraTS" or subject[:5]=="Brats"]

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
def interpolate_2D_image(img,desired_shape):


    x = np.linspace(0, desired_shape[1], img.shape[1])
    y = np.linspace(0,  desired_shape[0], img.shape[0])

    # Make the interpolator function.
    f = interpolate.interp2d(x, y, img, kind='linear')

    # Construct the new coordinate arrays.
    x_new = np.arange(0, desired_shape[1])
    y_new = np.arange(0, desired_shape[0])

    # Do the interpolation.
    new_img = f(x_new, y_new)

    return(new_img)




class TestBrats(BratsDataset):
    def __init__(self, config, subject_list):
        super(TestBrats, self).__init__(config, "valid")
        self.subject_list = subject_list

                





if __name__ == "__main__":
    from configuration import configuration
    
    project_name = 'basline'
    model_name = 'Baseline'
    debug_mode = True
    config = configuration(project_name, model_name, debug_mode, fold=2)
    prepare_data(config)
    """
    n_fold_split(config)
    select_slices_from_3D_images(config)
    mode =  "train"
    print(read_indices_2D_images_file(os.path.join(config['data_split_dir'], 
                                        '{mode}_set_fold_{fold}_indices_2D_images.txt'.format(mode = mode, fold = config['fold']))))
    """




























