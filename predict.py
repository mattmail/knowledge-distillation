import os
import torch
from models.HUnet import HUNetv3, UNet, HUNetv6
from dataset_brats import prepare_data
from loss_func import TotalLoss
from tools.plot import plot_nib



def predict(model_name, checkpoint="Checkpoint_best_fold_1.pkl"):
    checkpoint_path = os.path.join('../', 'result', model_name, 'pkl', checkpoint)
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']

    config['skip_connection'] = True

    if config['Model'] in ['Baseline', 'Teacher']:
        net = UNet(config).to(config['device'])
    elif config['Model'] == 'HUNetv3':
        net = HUNetv3(config).to(config['device'])
    elif config['Model'] == 'HUNetv6':
        net = HUNetv6(config).to(config['device'])
    else:
        assert "Wrong model name."

    net.load_state_dict(checkpoint['model_state_dict'])
    criterion = TotalLoss(config)
    config['debug_mode'] = True
    train_loader, valid_loader = prepare_data(config, shuffle=False)

    for i, data in enumerate(valid_loader, 0):
        imgs, labels = data
        inputs = [img.to(config['device']) for img in imgs]
        labels = labels.to(config['device'])

        if config['Model'] == 'Baseline':
            y_pred = net(inputs, 'source')
            loss_dict, dice_dict = criterion(y_pred, labels, similarity=False)
            # print(y_pred[0][0].shape)
            print(dice_dict)
            plot_nib(y_pred[0][0], "sample%d_pr" % i, config, channel_name=['ET', 'TC', 'WT'])
            plot_nib(labels[0], "sample%d_gt" % i, config, channel_name=['ET', 'TC', 'WT'])
            plot_nib(inputs, 'sample%d_' % i, config, channel_name=['T1', 'T1ce', 'T2', 'Flair'])

        else:
            y_pred, features = net(inputs, 'both')
            loss_dict, dice_dict = criterion(y_pred, labels, similarity=True, features=features)
            plot_nib(y_pred[0], "Sample%d" % i, config, channel_name=['ET', 'TC', 'WT'])