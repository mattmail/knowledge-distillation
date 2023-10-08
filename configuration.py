import os
import time
import random
import torch
import numpy as np
from tools.log import create_log


def configuration(project_name, model_name, debug_mode, fold=None, parallel=True, 
                            img_dim="2D", create_lg=True,mod=[1] ):

    # Global settings
    config = dict()
    config['debug_mode'] = debug_mode
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['workspace_root'] = "../"
    config['seed'] = 1
    random.seed(config["seed"])
    np.random.seed(config['seed'])

    # Model settings
    config['Model'] = model_name
    config['skip_connection'] = 4
    config['train_mode'] = 'Teacher_Student'                           # 'Teacher_Student': Train the source branch first, then freeze parameters and train the target branch;
                                                                # 'Mutually': Update both branches every iteration
    if config['Model'] in ['Baseline', 'Teacher']:
        config['train_mode'] = 'Mutually'

    if config['train_mode'] == 'Mutually':
        config['pretrained_teacher'] = None
    else:
        if fold == 1:
            #config['pretrained_teacher'] = 'teacher_2018_t1ce_full_fold1_0126_1859'
            config['pretrained_teacher'] = 'teacher_new_architecture_adam_t1ce_full_fold1_0207_1741'
            # config['pretrained_teacher'] = 'HUNetv3data_dir_t1ce_sc4_fold1_0202_0400'
        elif fold == 2:
            # config['pretrained_teacher'] = 'teacher_t1ce_fold2_0226_1502'
            # config['pretrained_teacher'] = 'HUNetv3_t1ce_sc4_fold2_0202_0400'
            config['pretrained_teacher'] = 'teacher_2018_t1ce_full_fold2_0126_1859'
        else:
            config['pretrained_teacher'] = 'teacher_2018_t1ce_full_fold3_0126_1900'
            # config['pretrained_teacher'] = 'HUNetv3_t1ce_sc4_fold3_0202_0401'
            # config['pretrained_teacher'] = 'v6_t1ce_fold3_0303_0954'
    #config['pretrained_teacher'] = 'teacher_t1ce_fold1_0226_1501'
    config['pretrained_student'] = None

    config['end_layer'] = 'softmax'
    config['softmax'] = True
    config['temperature'] = 5

    # Dataset settings
    config['dataset'] = 'BraTS_2018'  # 'BraTS_2018' or 'BraTS_2019'
    config['use_half'] = False
    config['use_quarter'] = False
    config['data_split'] = 'cross_validation' if config['pretrained_teacher'] is None else 'same_as_teacher'
    config['num_fold'] = 3
    config['conv_bias'] = True

    # train 3 folds at same time
    config['parallel'] = parallel
    if config['parallel']:
        config['fold'] = fold

    # Modalities
    config['modality_list'] = ['t1', 't1ce', 't2', 'flair']
    if model_name == 'Baseline':
        config['source_modality'] = mod
    elif model_name == 'Teacher':
        config['source_modality'] = [0, 1, 2, 3]
    else:
        config['source_modality'] = [0, 1, 2, 3]
    config['target_modality'] = mod

    config['num_source_modality'] = len(config['source_modality'])
    config['num_target_modality'] = len(config['target_modality'])
    
    config['train_val_split_ratio'] = 0.7 
    # image size and cropping
    config['crop_mode'] = "random"
    # Image dimension 
    config["img_dim"] = "3D"
    config["selected_slices_number"] = 30
    # this bbox is the smallest bbox that contain all the 3D brains 
    # in Brats2020.  
    config["void_crop_bbox"] = (-43, 41, -17, 29, -6, 0)  # (Xmax,Xmin, Ymax,Ymin,Zmax,Zmin ) 
    config["3D_img_size_after_cropping"] = [80,80,80]
    #print("We are working with {} images ".format(config["img_dim"]))
    if config["img_dim"] == "2D":
        config['generate_2D_data'] = False
        config['img_size'] = [192, 192]
        config["ratio_zero_pixels_2D_image"] = 0.97      
    elif config["img_dim"] == "3D": 
        #config['img_size'] = [128, 128, 128]
        config['crop_the_void_3D_img'] = True 
        config['img_size'] = [80,80,80]
    else:
        raise Exception("image dimension must be 3D or 2D")    
    config['half_size'] = False #or config['debug_mode']



    # data preprocessing and augmentation
    config['data_preprocessing'] = True
    config['data_augmentation'] = False
    config['upper_clip'] = 8000
    config['random_mirror_flip'] = False

    # training configurations
    config['num_epoch'] = 20 if config['debug_mode'] else 1000
    config['batch_size'] = 2 if not config['debug_mode'] else 1
    config['lr'] = 1e-3
    config['weight_decay'] = 1e-5

    # Loss function settings
    config['train_loss_label'] = ['total_loss']
    config['valid_loss_label'] = ['total_loss']
    config['dice_label'] = ['Dice_ET', 'Dice_TC', 'Dice_WT']

    # Loss function and metrics
    ## Segmentation loss functions
    # soft dice loss
    config['lambda'] = 0.75
    config['dice_loss'] = True
    if config['dice_loss']:
        config['generalised_dice'] = False
        config['dice_epsilon'] = 1e-10
        config['dice_weight'] = 1. # - config['lambda']
        config['train_loss_label'].append('dice_loss_source')
        config['train_loss_label'].append('dice_loss_target')
        config['valid_loss_label'].append('dice_loss_target')
        config['dice_square'] = False

    # binary cross entropy
    config['bce_loss'] = False
    if config['bce_loss']:
        if config["img_dim"] == "2D":
            config['bce_weight'] = 10*(1 - config['lambda'])
        else: 
            config['bce_weight'] = 1 - config['lambda']   
        config['train_loss_label'].append('bce_loss_source')
        config['train_loss_label'].append('bce_loss_target')
        config['valid_loss_label'].append('bce_loss_target')

    config['ce_loss'] = True
    if config['ce_loss']:
        if config["img_dim"] == "2D":
            config['ce_weight'] = 10 * (1 - config['lambda'])
        else:
            config['ce_weight'] = 1. # - config['lambda']
        config['train_loss_label'].append('ce_loss_source')
        config['train_loss_label'].append('ce_loss_target')
        config['valid_loss_label'].append('ce_loss_target')

    config['focal_loss'] = False
    if config['focal_loss']:
        config['focal_gamma'] = 2
        config['focal_weight'] = 1e-5
        config['train_loss_label'].append('focal_loss_source')
        config['train_loss_label'].append('focal_loss_target')
        config['valid_loss_label'].append('focal_loss_target')

    ## Similarity loss functions
    # Kullback Leibler divergence
    config['KL_loss'] = False and (config['Model'] != 'Baseline')
    if config['KL_loss']:
        config['KL_weight'] = 10
        config['train_loss_label'].append('KL_loss')

    config['Att_loss'] = False and (config['Model'] != 'Baseline')
    if config['Att_loss']:
        config['Att_weight'] = 10
        config['train_loss_label'].append('Att_loss')

    config['kd_loss'] = True and (config['Model'] != 'Baseline')
    if config['kd_loss']:
        config['kd_weight'] = 0.5 #config['lambda']
        config['train_loss_label'].append('kd_dice_loss')
        config['train_loss_label'].append('kd_ce_loss')

    config['contrastive_loss'] = True and (config['Model'] != 'Baseline')
    if config['contrastive_loss']:
        config['contrastive_weight'] = 0.5
        config['train_loss_label'].append('contrastive_loss')

    # save settings
    config['save_epoch'] = 100 if not config['debug_mode'] else 1
    config['init_epoch'] = 1
    config['return_filename'] = False

    # Project settings
    config['project_name'] = project_name
    for i in config['target_modality']:
        config['project_name'] += '_' + config['modality_list'][i]
    if not config['half_size']:
        config['project_name'] += '_full'
    if config['debug_mode']:
        config['project_name'] += '_debug'
    if config['parallel']:
        config['project_name'] += '_fold%d' % config['fold']
    if config['use_half']:
        config['project_name'] += "_half"
    if config['use_quarter']:
        config['project_name'] += "_quarter"

    config['description'] = ''
    config['model_name'] = config['project_name'] + "_" + time.strftime("%m%d_%H%M", time.localtime())
    if create_lg: 
        config['result_path'] = create_log(config)
        config['data_split_dir'] = os.path.join(config['result_path'], 'data_split')

    return config
