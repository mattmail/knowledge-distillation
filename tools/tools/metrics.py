import numpy as np
from dataset_brats import seg_label
import torch
from math import ceil


def filter_linear(pr):
    pr[np.where(pr > 1)] = 1
    pr[np.where(pr < 0)] = 0
    return pr


def filter_binary(pr):
    pr[np.where(pr > 0.5)] = 1
    pr[np.where(pr <= 0.5)] = 0
    return pr


def label_trans(pr_torch, gt_torch, need_detach=False):
    pr = pr_torch.cpu().detach().numpy()
    if need_detach:
        gt = gt_torch.cpu().detach().numpy()
    else:
        gt = gt_torch.cpu().numpy()
    argmax = np.argmax(pr, axis=0)
    pred = argmax + 1 * (argmax == 3)
    pr = seg_label(pred)
    gtmap = gt[1] * 1 + gt[2] * 2 + gt[3] * 4
    gt = seg_label(gtmap)
    return pr, gt


def cal_dice_score(pr_torch, gt_torch, need_detach=False, to_np=True, eps = 1e-5):

    if to_np:
        # dice score without square
        pr = pr_torch.cpu().detach().numpy()
        if need_detach:
            gt = gt_torch.cpu().detach().numpy()
        else:
            gt = gt_torch.cpu().numpy()
    else:
        pr = pr_torch
        gt = gt_torch

    # Some LGG samples don't have a Enhanced Tumor
    if np.sum(gt) == 0:
        if np.sum(pr) == 0:
            return np.asarray([1])
        else:
            return np.asarray([0])
        '''
        if np.sum(pr) == 0:
            return np.asarray([1, 1, 1, 1, 1, 1])
        else:
            return np.asarray([0, 0, 0, 0, 0, 0])
        '''
    else:
        '''
        gt_pr = np.sum(pr * gt)
        pr_pr = np.sum(pr * pr)
        gt_gt = np.sum(gt * gt)

        # dice score with square
        dice0 = 2.0 * gt_pr / (pr_pr + gt_gt)

        # dice score without square
        dice1 = 2.0 * gt_pr / (np.sum(pr) + np.sum(gt))

        # linear filter for prediction
        pr_ = filter_linear(pr.copy())
        gt_pr_ = np.sum(pr_ * gt)
        pr_pr_ = np.sum(pr_ * pr_)

        # dice score with square + filter
        dice2 = 2.0 * gt_pr_ / (pr_pr_ + gt_gt)

        # dice score without square + filter
        dice3 = 2.0 * gt_pr_ / (np.sum(pr_) + np.sum(gt))
        '''

        # binary filter for prediction
        pr_bin = filter_binary(pr.copy())
        gt_pr_bin = np.sum(pr_bin * gt)

        '''
        pr_pr_bin = np.sum(pr_bin * pr_bin)

        # dice score with square + binary
        dice4 = 2.0 * gt_pr_bin / (pr_pr_bin + gt_gt)
        '''

        # dice score without square + binary
        dice5 = (2.0 * gt_pr_bin + eps) / (np.sum(pr_bin) + np.sum(gt)+ eps)

        # return np.asarray([dice0, dice1, dice2, dice3, dice4, dice5])
        return np.asarray([dice5])

def random_crop_test(model, inputs, mode, config):
    bts_size,_,h, w, d = inputs[0].shape
    HEIGHT, WIDTH, DEPTH = config['img_size']
    NUM_CLS = 4
    overlap_perc_h, overlap_perc_w, overlap_perc_d = 0.5, 0.5, 0.5

    h_cnt = ceil((h - HEIGHT) / (HEIGHT * (1 - overlap_perc_h)))
    h_idx_list = range(0, h_cnt)
    h_idx_list = [h_idx * int(HEIGHT * (1 - overlap_perc_h)) for h_idx in h_idx_list]
    h_idx_list.append(h - HEIGHT)

    w_cnt = ceil((w - WIDTH) / (WIDTH * (1 - overlap_perc_w)))
    w_idx_list = range(0, w_cnt)
    w_idx_list = [w_idx * int(WIDTH * (1 - overlap_perc_w)) for w_idx in w_idx_list]
    w_idx_list.append(w - WIDTH)

    d_cnt = ceil((d - DEPTH) / (DEPTH * (1 - overlap_perc_d)))
    d_idx_list = range(0, d_cnt)
    d_idx_list = [d_idx * int(DEPTH * (1 - overlap_perc_d)) for d_idx in d_idx_list]
    d_idx_list.append(d - DEPTH)

    pred_whole = torch.zeros((bts_size, NUM_CLS, h, w, d)).to(config['device'])
    avg_whole = torch.zeros((bts_size,NUM_CLS, h, w, d)).to(config['device'])
    avg_block = torch.ones((bts_size,NUM_CLS, HEIGHT, WIDTH, DEPTH)).to(config['device'])
    for d_idx in d_idx_list:
        for w_idx in w_idx_list:
            for h_idx in h_idx_list:
                data_list_crop = []
                for d_iter in range(0, len(inputs)):
                    data_list_crop.append(inputs[d_iter][:,:,h_idx:h_idx + HEIGHT, w_idx:w_idx + WIDTH,
                                          d_idx:d_idx + DEPTH])

                seg_pred_val = model(data_list_crop, mode)
                pred_whole[:,:,h_idx:h_idx + HEIGHT, w_idx:w_idx + WIDTH, d_idx:d_idx + DEPTH] += seg_pred_val[0]
                avg_whole[:,:,h_idx:h_idx + HEIGHT, w_idx:w_idx + WIDTH, d_idx:d_idx + DEPTH] += avg_block

    return [pred_whole / avg_whole]
