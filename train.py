import os

import matplotlib.pyplot as plt

from tools.log import add_log
from tools.load_model import load_model_
from tools.save_model import save_model
from tools.history import History, History_branch
from tools.metrics import random_crop_test
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch
from IPython import display
#from tensorboardX import SummaryWriter
import torchio as tio
def convert_labels(seg):
    label_0 = 1 * (seg == 0)
    label_1 = 1 * (seg == 1)
    label_2 = 1 * (seg == 2)
    label_4 = 1 * (seg == 4)
    labels = torch.cat([label_0, label_1, label_2, label_4], dim=1)
    return labels

def train_model(model, criterion,  optimizer, train_loader, valid_loader, config, mode, scheduler):
    print("==> Start training...")
    lr = config['lr']
    max_dice_avg = 0.
    tensorboard = SummaryWriter(config['result_path'])
    valid_every_n_epoch = 1 if config['debug_mode'] else 20

    train_loss = History(config['train_loss_label'], 'Train')
    valid_loss = History(config['valid_loss_label'], 'Valid')
    valid_dice = History(config['dice_label'], "Valid")
    if mode == 'source' or config['Model'] in ['Baseline', 'Teacher']:
        train_branch = 'source'
        valid_branch = 'source'
        train_dice = History(config['dice_label'], "Train")
    elif mode == 'both':
        train_branch = 'both'
        valid_branch = 'target'
        train_dice = History_branch(config['dice_label'], "Train")
    else:
        assert "wrong train mode for model training."

    for epoch in range(config['init_epoch'], config['num_epoch']+1):
        add_log(config, "[Fold: %d / %d   Epoch: %d / %d]" %
                (config['fold'], config['num_fold'], epoch, config['num_epoch']))

        train_loss, train_dice = train_epoch(model, criterion, optimizer, train_loader, config, train_loss, train_dice, train_branch)
        if epoch % valid_every_n_epoch == 0:
            valid_loss, valid_dice = valid_epoch(model, criterion, valid_loader, config, valid_loss, valid_dice, valid_branch)
            scheduler.step(valid_loss.dict['dice_loss_target'][-1])

        if train_branch == 'both':
            tensorboard.add_scalars(config['model_name'] + '_train/fold_%d/source_dice' % (config['fold']),
                                    train_dice.tail(0)[0], epoch)
            tensorboard.add_scalars(config['model_name'] + '_train/fold_%d/target_dice' % (config['fold']),
                                    train_dice.tail(0)[1], epoch)
        else:
            tensorboard.add_scalars(config['model_name'] + '_train/fold_%d/%s_dice' % (config['fold'], train_branch),
                                train_dice.tail(0), epoch)
        tensorboard.add_scalars(config['model_name'] + '_train/fold_%d/loss' % (config['fold']),
                                train_loss.tail(), epoch)
        if epoch % valid_every_n_epoch == 0:
            tensorboard.add_scalars(config['model_name'] + '_valid/fold_%d/%s_dice' % (config['fold'], valid_branch),
                                valid_dice.tail(0), epoch)
            tensorboard.add_scalars(config['model_name'] + '_valid/fold_%d/loss' % (config['fold']),
                                valid_loss.tail(), epoch)

        if epoch % valid_every_n_epoch == 0:
            current_avg = (valid_dice.dict['Dice_ET'][-1] + valid_dice.dict['Dice_TC'][-1] + valid_dice.dict['Dice_WT'][-1])/3.0
            if max_dice_avg < current_avg:
                max_dice_avg = current_avg
                add_log(config, "Saving best model for epoch %d" % epoch)
                save_model(model, optimizer, config, epoch, best=True)
                with open(os.path.join(config['result_path'], "best_score.txt"), 'w') as f:
                    f.write("Epoch %d - ET: %f, TC: %f, WT: %f" %(epoch, valid_dice.dict['Dice_ET'][-1], valid_dice.dict['Dice_TC'][-1], valid_dice.dict['Dice_WT'][-1]))

        if scheduler.num_bad_epochs == 5:
            model, checkpoint = load_model_(config, model)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for param in optimizer.param_groups:
                lr *= 0.2
                param['lr'] = lr
                add_log(config, "reducing learning rate to %f" % param['lr'])
            scheduler.num_bad_epochs = 0

        if lr < 1e-8:
            add_log(config, "early stop...")
            break

        if epoch % config['save_epoch'] == 0:
            add_log(config, "Saving model for epoch %d" % epoch)
            save_model(model, optimizer, config, epoch)


def train_epoch(model, criterion,  optimizer, train_loader, config, train_loss, train_dice, branch):
    model.train()
    epoch_loss = History(config['train_loss_label'], 'train')
    if branch == 'source':
        epoch_dice = History(config['dice_label'], 'train')
    else:
        epoch_dice = History_branch(config['dice_label'], 'train')

    progress_bar = tqdm(total=len(train_loader))

    for i, data in enumerate(train_loader, 0):
        if config['debug_mode']:
            if i >= 5:
                break
        #imgs, labels = data
        imgs = [data['t1'][tio.DATA], data['t1ce'][tio.DATA], data['t2'][tio.DATA], data['flair'][tio.DATA]]
        labels = convert_labels(data['seg'][tio.DATA])
        inputs = [img.to(config['device']) for img in imgs]
        labels = labels.to(config['device'])
        if branch == 'source':
            y_pred = model(inputs, 'source')
            loss_dict, dice_dict = criterion(y_pred, labels, similarity=False)
        else:
            y_pred, soft, de,  features = model(inputs, 'both', temperature=True)
            loss_dict, dice_dict = criterion(y_pred, labels, similarity=True, soft=soft, de=de,  features=features)

        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        optimizer.step()
        epoch_loss.update(loss_dict, need_item=True)
        epoch_dice.update(dice_dict, need_item=False)
        progress_bar.update(1)
    progress_bar.close()

    avg_loss = epoch_loss.average()
    avg_dice = epoch_dice.average()
    epoch_loss.log_avg(config)
    epoch_dice.log_avg(config, format='%s\t')
    train_loss.update(avg_loss)
    train_dice.update(avg_dice)

    return train_loss, train_dice


def valid_epoch(model, criterion, valid_loader, config, valid_loss, valid_dice, branch):
    model.eval()
    epoch_loss = History(config['valid_loss_label'], 'valid')
    epoch_dice = History(config['dice_label'], 'valid')
    progress_bar = tqdm(total=len(valid_loader))
    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            if config['debug_mode']:
                if i >= 5:
                    break
            # imgs, labels = data
            imgs = [data['t1'][tio.DATA], data['t1ce'][tio.DATA], data['t2'][tio.DATA], data['flair'][tio.DATA]]
            labels = convert_labels(data['seg'][tio.DATA])
            inputs = [img.to(config['device']) for img in imgs]
            labels = labels.to(config['device'])

            if config['crop_mode'] == "center":
                y_pred = model(inputs, branch)
            elif config['crop_mode'] == "random":
                y_pred = random_crop_test(model, inputs, branch, config)
            else:
                assert "error crop mode is unknown"
            loss_dict, dice_dict = criterion(y_pred, labels, similarity=False)

            epoch_loss.update(loss_dict, need_item=True)
            epoch_dice.update(dice_dict, need_item=False)
            progress_bar.update(1)
        progress_bar.close()

        avg_loss = epoch_loss.average()
        avg_dice = epoch_dice.average()
        epoch_loss.log_avg(config)
        epoch_dice.log_avg(config, format='%s\t')
        valid_loss.update(avg_loss)
        valid_dice.update(avg_dice)

    return valid_loss, valid_dice
