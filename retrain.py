import argparse
import os

import torch
import torch.optim as optim
from models.HUnet import UNet, HUNetv4
from models.PMKL_model import Baseline, Teacher, KDNet, KDNet_share
from dataset_brats import prepare_data
from train import train_model
from loss_func import TotalLoss
from tools.load_model import load_teacher, load_teacher_PMKL
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tools.save_model import freeze_model, check_freeze, save_model_source, freeze_teacher_PMKL
from tools.log import add_log
from itertools import chain

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--proj_name", default="teacher")
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    project_name = args.proj_name
    debug_mode = args.debug
    checkpoint_path = os.path.join(os.environ['HOME'], "knowledge-distillation", 'results', project_name, 'pkl', "Checkpoint_best_fold_1.pkl")
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']

    if config['Model'] in ['Baseline', 'Teacher']:
        net = UNet(config).to(config['device'])
    elif config['Model'] == 'HUNetv4':
        net = HUNetv4(config).to(config['device'])
    else:
        assert "Wrong model name."

    net.load_state_dict(checkpoint['model_state_dict'])
    criterion = TotalLoss(config)
    config['debug_mode'] = debug_mode
    config['batch_size'] = 4
    train_loader, valid_loader = prepare_data(config)

    #To train the teacher or baseline alone
    if config['train_mode'] == 'Mutually':
        if config['pretrained_student'] is not None:
            net, checkpoint = load_teacher(config, net, from_UNet=False, load_student=True)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config['lr'],
                               weight_decay=config['weight_decay'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=51, factor=0.2, verbose=True)

        train_model(net, criterion, optimizer, train_loader, valid_loader, config, 'both', scheduler)

    #to train the student
    elif config['train_mode'] == 'Teacher_Student':
        if config['pretrained_teacher'] is not None and config['pretrained_student'] is None:
            #net, checkpoint = load_teacher(config, net, from_UNet=True, for_HUNetv4=(config['Model']=='HUNetv4'))
            freeze_model(config, net, branch='source', freeze_share=False)
            check_freeze(config, net)
            if config["contrastive_loss"]:
                optimizer_target = optim.Adam(chain(filter(lambda p: p.requires_grad, net.parameters()),
                                                    filter(lambda p: p.requires_grad,
                                                           criterion.ct_loss.mlp.parameters())), lr=config['lr'],
                                              weight_decay=config['weight_decay'])
            else:
                optimizer_target = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config['lr'],
                                              weight_decay=config['weight_decay'])
            optimizer_target.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler = ReduceLROnPlateau(optimizer_target, 'min', patience=51, factor=0.2, verbose=True)

            add_log(config, "Start training the target branch...")
            train_model(net, criterion, optimizer_target, train_loader, valid_loader, config, 'both', scheduler)
        elif config['pretrained_student'] is None:
            freeze_model(config, net, branch='source', freeze_share=False)
            check_freeze(config, net)
            if config["contrastive_loss"]:
                optimizer_source = optim.Adam(chain(filter(lambda p: p.requires_grad, net.parameters()),
                                                    filter(lambda p: p.requires_grad,
                                                           criterion.ct_loss.mlp.parameters())), lr=config['lr'],
                                              weight_decay=config['weight_decay'])
            else:
                optimizer_source = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config['lr'],
                                              weight_decay=config['weight_decay'])
            optimizer_source.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler = ReduceLROnPlateau(optimizer_source, 'min', patience=51, factor=0.2, verbose=True)

            add_log(config, "Start training the target branch...")
            train_model(net, criterion, optimizer_source, train_loader, valid_loader, config, 'source', scheduler)

        elif config['pretrained_student'] is not None:
            net, checkpoint = load_teacher(config, net, from_UNet=False, load_student=True)
            freeze_model(config, net, branch='source', freeze_share=False)
            check_freeze(config, net)

            if config["contrastive_loss"]:
                optimizer = optim.Adam(chain(filter(lambda p: p.requires_grad, net.parameters()),
                                                    filter(lambda p: p.requires_grad,
                                                           criterion.ct_loss.mlp.parameters())), lr=config['lr'],
                                              weight_decay=config['weight_decay'])
            else:
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config['lr'],
                                              weight_decay=config['weight_decay'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=51, factor=0.2, verbose=True)

            add_log(config, "Start training the target branch...")
            train_model(net, criterion, optimizer, train_loader, valid_loader, config, 'both', scheduler)

    return config


if __name__ == '__main__':
    main()






