import os
import torch
from models.HUnet import HUNetv4, HUNetv6
from models.HUnet import UNet
from dataset_brats import prepare_data
from loss_func import TotalLoss
from tools.plot import plot_nib
import numpy as np



def predict(model_name_b, model_name_s, checkpoint="Checkpoint_best_fold_1.pkl"):
    checkpoint_path_b = os.path.join('../', 'result', model_name_b, 'pkl', checkpoint)
    checkpoint_b = torch.load(checkpoint_path_b)
    config_b = checkpoint_b['config']

    checkpoint_path_s = os.path.join('../', 'result', model_name_s, 'pkl', checkpoint)
    checkpoint_s = torch.load(checkpoint_path_s)
    config_s = checkpoint_s['config']

    if config_b['Model'] in ['Baseline', 'Teacher']:
        net_b = UNet(config_b).to(config_b['device'])
    elif config_b['Model'] == 'HUNetv4':
        net_b = HUNetv4(config_b).to(config_b['device'])
    else:
        assert "Wrong model name."

    if config_s['Model'] in ['Baseline', 'Teacher']:
        net_s = UNet(config_s).to(config_b['device'])
    elif config_s['Model'] == 'HUNetv4':
        net_s = HUNetv4(config_s).to(config_b['device'])
    else:
        assert "Wrong model name."
    print(checkpoint_b['model_state_dict'].keys())
    net_b.load_state_dict(checkpoint_b['model_state_dict'])
    criterion_b = TotalLoss(config_b)
    config_b['debug_mode'] = True
    train_loader, valid_loader = prepare_data(config_b, shuffle=False)

    net_s.load_state_dict(checkpoint_s['model_state_dict'])
    criterion_s = TotalLoss(config_s)
    config_s['debug_mode'] = True
    disc = []
    for i, data in enumerate(valid_loader, 0):
        imgs, labels = data
        inputs = [img.to(config_b['device']) for img in imgs]
        labels = labels.to(config_b['device'])

        y_pred = net_b(inputs, 'source')
        loss_dict, dice_dict_b = criterion(y_pred, labels, similarity=False)
        # print(y_pred[0][0].shape)
        print(dice_dict)

        y_pred, features = net_s(inputs, 'both')
        loss_dict, dice_dict_s = criterion(y_pred, labels, similarity=True, features=features)
        disc.append(dice_dict_s['dice_ET'] - dice_dict_b['dice_ET']+dice_dict_s['dice_TC'] - dice_dict_b['dice_TC']+dice_dict_s['dice_WT'] - dice_dict_b['dice_WT'])
        #plot_nib(y_pred[0], "Sample%d" % i, config, channel_name=['ET', 'TC', 'WT'])
    disc = np.array(disc)
    print(np.max(disc))
    print(np.argmax(disc))
