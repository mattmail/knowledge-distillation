import torch
import os
import shutil
from tools.log import add_log


def save_model(model, optimizer, config, epoch, best=False):
    checkpoint = {
        'epoch': epoch,
        'config': config,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    pkl_dir = os.path.join(config['result_path'], "pkl")
    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)
    if best:
        pkl_path = os.path.join(pkl_dir, "Checkpoint_best_fold_%d.pkl" % config['fold'])
    else:
        pkl_path = os.path.join(pkl_dir, "Checkpoint_fold_%d.pkl" % config['fold'])
    torch.save(checkpoint, pkl_path)


def save_model_source(config):
    pkl_dir = os.path.join(config['result_path'], "pkl")
    pkl_path_from = os.path.join(pkl_dir, "Checkpoint_best_fold_%d.pkl" % config['fold'])
    pkl_path_to = os.path.join(pkl_dir, "Checkpoint_best_fold_%d_source.pkl" % config['fold'])
    if os.path.exists(pkl_path_to):
        print("Best source model already exists..")
    else:
        shutil.copy(pkl_path_from, pkl_path_to)
    pkl_path_from = os.path.join(pkl_dir, "Checkpoint_fold_%d.pkl" % config['fold'])
    pkl_path_to = os.path.join(pkl_dir, "Checkpoint_fold_%d_source.pkl" % config['fold'])
    if os.path.exists(pkl_path_to):
        print("Source model already exists..")
    else:
        shutil.copy(pkl_path_from, pkl_path_to)


def freeze_model(config, model, branch, freeze_share=True):
    add_log(config, "Freezing the source branch...")
    if branch == 'target':
        freeze_branch = ['encoder_target', 'decoder_target']
    else:
        freeze_branch = ['encoder_source', 'decoder_source', 'bottle_neck_source']

    if freeze_share:
        freeze_branch += ['decoder_share', 'encoder_share', 'bottle_neck_share']

    for name, child in model.named_children():
        if name in freeze_branch:
            dfs_helper(child, False)
        else:
            dfs_helper(child, True)

def freeze_teacher_PMKL(config, model, share=False):
    add_log(config, "Freezing the source branch...")
    freeze_branch = ['encoder1','encoder3','encoder4', "fusion1","fusion2","fusion3","fusion4", "decoder"]
    if not share:
        freeze_branch.append("encoder2")

    for name, child in model.named_children():
        if name in freeze_branch:
            dfs_helper(child, False)
        else:
            dfs_helper(child, True)


def dfs_helper(model, requires_grad=True):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = requires_grad
            if not requires_grad:
                param.detach_()
        dfs_helper(child, requires_grad)


def check_freeze(config, model):
    for name, child in model.named_children():
        flag = True
        for param in child.parameters():
            flag = flag & param.requires_grad
        add_log(config, "%s: %s" % (name, flag))