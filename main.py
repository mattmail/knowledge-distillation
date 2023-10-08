import argparse
import torch.optim as optim
from models.HUnet import UNet, HUNetv4
from models.PMKL_model import Baseline, Teacher, KDNet, KDNet_share
#from dataset_brats import prepare_data, n_fold_split, copy_data_split
from brats_torchio import prepare_data
from train import train_model
from loss_func import TotalLoss
from tools.load_model import load_teacher, load_teacher_PMKL
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tools.save_model import freeze_model, check_freeze, save_model_source, freeze_teacher_PMKL
from tools.log import add_log
from configuration import configuration
from itertools import chain

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--proj_name", default="teacher")
    parser.add_argument("-m", "--model_name", default="Teacher")
    parser.add_argument("-f", "--fold", help='1, 2, or 3', default=1)
    parser.add_argument("--debug", action='store_true') 
    parser.add_argument("--mod", nargs = "*", type=int, default=[1])
    args = parser.parse_args()
    project_name = args.proj_name
    model_name = args.model_name
    debug_mode = args.debug
    mod = args.mod
    img_dim = "3D"
    fold= int(args.fold)
    config = configuration(project_name, model_name, debug_mode, fold=fold, img_dim=img_dim, create_lg=True, mod=mod)
    use_normal = True
    if config['parallel']:
        train(config, use_normal=use_normal)
    else:
        for fold in range(1, config['num_fold']+1):
            config['fold'] = fold
            train(config, use_normal=use_normal)
    add_log(config, "All complete...")


def train(config, load_model=False, checkpoint=None, use_normal=True):
    if use_normal:
        if config['Model'] in ['Baseline', 'Teacher']:
            net = UNet(config)
        elif config['Model'] == 'HUNetv4':
            net = HUNetv4(config)
        else:
            assert "Wrong model name."
    else:
        if config['Model'] == 'Baseline':
            net = Baseline(config)
        elif config['Model'] == 'Teacher':
            net = Teacher(config)
        elif config['Model'] == 'KDNet':
            net = KDNet(config)
        elif config['Model'] == 'KDNet_share':
            net = KDNet_share(config)
    net = net.to(config['device'])
    criterion = TotalLoss(config)
    train_loader, valid_loader, _ = prepare_data(config)

    #To train the teacher or baseline alone
    if config['train_mode'] == 'Mutually':
        if config['pretrained_student'] is not None:
            net, checkpoint = load_teacher(config, net, from_UNet=False, load_student=True)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config['lr'],
                               weight_decay=config['weight_decay'])
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=51, factor=0.2, verbose=True)

        train_model(net, criterion, optimizer, train_loader, valid_loader, config, 'both', scheduler)

    #to train the student
    elif config['train_mode'] == 'Teacher_Student':
        if use_normal:
            if config['pretrained_teacher'] is not None and config['pretrained_student'] is None:
                net, checkpoint = load_teacher(config, net, from_UNet=True, for_HUNetv4=(config['Model']=='HUNetv4'))
                freeze_model(config, net, branch='source', freeze_share=True)
                check_freeze(config, net)
                if config["contrastive_loss"]:
                    optimizer_target = optim.Adam(chain(filter(lambda p: p.requires_grad, net.parameters()),
                                                        filter(lambda p: p.requires_grad,
                                                               criterion.ct_loss.mlp.parameters())), lr=config['lr'],
                                                  weight_decay=config['weight_decay'])
                else:
                    optimizer_target = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config['lr'],
                                                  weight_decay=config['weight_decay'])

                scheduler = ReduceLROnPlateau(optimizer_target, 'min', patience=51, factor=0.2, verbose=True)

                add_log(config, "Start training the target branch...")
                train_model(net, criterion, optimizer_target, train_loader, valid_loader, config, 'both', scheduler)
            elif config['pretrained_student'] is None:
                freeze_model(config, net, branch='target', freeze_share=False)
                check_freeze(config, net)
                if config["contrastive_loss"]:
                    optimizer_source = optim.Adam(chain(filter(lambda p: p.requires_grad, net.parameters()),
                                                        filter(lambda p: p.requires_grad,
                                                               criterion.ct_loss.mlp.parameters())), lr=config['lr'],
                                                  weight_decay=config['weight_decay'])
                else:
                    optimizer_source = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config['lr'],
                                                  weight_decay=config['weight_decay'])

                scheduler = ReduceLROnPlateau(optimizer_source, 'min', patience=51, factor=0.2, verbose=True)

                add_log(config, "Start training the target branch...")
                train_model(net, criterion, optimizer_source, train_loader, valid_loader, config, 'source', scheduler)

            elif config['pretrained_student'] is not None:
                net, checkpoint = load_teacher(config, net, from_UNet=False, load_student=True)
                freeze_model(config, net, branch='source', freeze_share=True)
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
        else:
            net, checkpoint = load_teacher_PMKL(config, net)
            if config['Model'].split('_')[-1] == "share":
                freeze_teacher_PMKL(config, net, share=True)
            else:
                freeze_teacher_PMKL(config, net)
            check_freeze(config, net)
            if config["contrastive_loss"]:
                optimizer_target = optim.Adam(chain(filter(lambda p: p.requires_grad, net.parameters()),
                                                    filter(lambda p: p.requires_grad,
                                                           criterion.ct_loss.mlp.parameters())), lr=config['lr'],
                                              weight_decay=config['weight_decay'])
            else:
                optimizer_target = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config['lr'],
                                              weight_decay=config['weight_decay'])

            scheduler = ReduceLROnPlateau(optimizer_target, 'min', patience=51, factor=0.2, verbose=True)

            add_log(config, "Start training the target branch...")
            train_model(net, criterion, optimizer_target, train_loader, valid_loader, config, 'both', scheduler)


    return config


if __name__ == '__main__':
    main()






