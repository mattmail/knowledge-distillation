import torch
from torch.nn.modules.loss import _Loss
import torch.nn as nn
from tools.metrics import cal_dice_score, label_trans
import numpy as np
from operator import mul
from functools import reduce
from skimage.metrics import hausdorff_distance


class KLLoss_vae(_Loss):
    """ KL divergence loss.
    A standard VAE penalty term, a KL divergence between the estimated normal distribution and a prior distribution
    N(0,1).
    """

    def __init__(self):
        super(KLLoss_vae, self).__init__()

    def forward(self, y_distr):
        mean = y_distr[:, : 128]
        logvar = y_distr[:, 128:]
        kl_loss = torch.sum(torch.mul(mean, mean) + torch.exp(logvar) - logvar - 1)
        return kl_loss


class KLLoss(_Loss):
    def __init__(self, config):
        super(KLLoss, self).__init__()
        self.batch_size = config['batch_size']
        self.epsilon = 1e-8

    def forward(self, tar, src):
        src_ = torch.sigmoid(src.view((self.batch_size, -1)))
        tar_ = torch.sigmoid(tar.view((self.batch_size, -1)))

        py = torch.div(tar_.t(), torch.sum(tar_, dim=1))
        px = torch.div(src_.t(), torch.sum(src_, dim=1))
        KL = torch.trace(torch.matmul(px.t(), torch.log(px / py + self.epsilon)))
        return KL


class ATTLoss(_Loss):
    def __init__(self, config):
        super(ATTLoss, self).__init__()
        self.batch_size = config['batch_size']

    def forward(self, tar, src):
        src = torch.sum(src ** 2, dim=1)
        tar = torch.sum(tar ** 2, dim=1)
        src_ = src.view((src.shape[0], -1))
        tar_ = tar.view((tar.shape[0], -1))

        tar_norm = torch.sqrt(torch.sum(tar_ ** 2, dim=1)).unsqueeze(1)
        src_norm = torch.sqrt(torch.sum(src_ ** 2, dim=1)).unsqueeze(1)
        return torch.sum(torch.sum((src_ / src_norm - tar_ / tar_norm) ** 2, dim=1).sqrt())


class MLPContrastive(nn.Module):
    def __init__(self, config):
        super(MLPContrastive, self).__init__()
        in_channels = int(reduce(mul, config['img_size'], 1) / (2**len(config['img_size'])) * 30)
        if config['half_size']:
            in_channels = int(in_channels/2**len(config['img_size']))
        self.pooling = nn.AvgPool3d(2)
        self.layer = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

    def forward(self, x):
        x = self.pooling(x)
        x = x.view((x.shape[0], -1))
        features = self.layer(x)
        norm = torch.sqrt(torch.sum(features ** 2, dim=-1, keepdim=True))
        return features / norm


class ContrastiveLoss(_Loss):
    def __init__(self, config):
        super(ContrastiveLoss, self).__init__()
        self.device = config["device"]
        self.mlp = MLPContrastive(config).to(self.device)

    def forward(self, student, teacher):
        #x_s = student.view((student.shape[0], -1))
        #x_t = teacher.view((teacher.shape[0], -1))
        feat_s = self.mlp(student)
        feat_t = self.mlp(teacher)
        loss = 0
        bs = student.shape[0]
        for i in range(bs):
            for j in range(bs):
                if i == j:
                    loss += torch.sum((feat_t[i] - feat_s[j]) ** 2)
                else:
                    dist = 1 - torch.sqrt(torch.sum((feat_t[i] - feat_s[j]) ** 2))
                    loss += torch.max(torch.zeros(1).to(self.device), dist.unsqueeze(0)).squeeze()**2
        return loss


class SoftDiceLoss(_Loss):
    def __init__(self, config):
        super(SoftDiceLoss, self).__init__()
        self.config = config

    def cal_dice_loss(self, pred, gt):
        eps = self.config['dice_epsilon']
        if self.config['dice_square']:
            gt_pr = torch.sum(torch.mul(pred, gt))
            gt_gt = torch.sum(torch.mul(gt, gt))
            pr_pr = torch.sum(torch.mul(pred, pred))
            dice_eps = (2. * gt_pr + eps) / (gt_gt + pr_pr + eps)
        else:
            tp = torch.sum(torch.mul(pred, gt))
            fp = torch.sum(torch.mul(pred, 1 - gt))
            fn = torch.sum(torch.mul(1 - pred, gt))
            dice_eps = (2. * tp + eps) / (2. * tp + fp + fn + eps)
        return 1 - dice_eps

    def forward(self, y_pred, y_gt, need_detach=False, only_WT=False):
        dice_score = dict()
        dice_score['Dice_ET'] = 0.0
        dice_score['Dice_TC'] = 0.0
        dice_score['Dice_WT'] = 0.0
        dice_loss = 0.0

        for i in range(y_gt.shape[0]):
            if self.config['end_layer'] == 'softmax':
                dice_loss_0 = self.cal_dice_loss(y_pred[i][0], y_gt[i][0])
                dice_loss_1 = self.cal_dice_loss(y_pred[i][1], y_gt[i][1])
                dice_loss_2 = self.cal_dice_loss(y_pred[i][2], y_gt[i][2])
                dice_loss_4 = self.cal_dice_loss(y_pred[i][3], y_gt[i][3])

                dice_loss += (dice_loss_0 + dice_loss_1 + dice_loss_2 + dice_loss_4) / 4.0

                pr, gt = label_trans(y_pred[i], y_gt[i], need_detach)

                if pr[0].sum() < 50:
                    pr[1][pr[0] == 1] = 1.
                    pr[0][pr[0] == 1.] = 0

                dice_score_ET = cal_dice_score(pr[0], gt[0], need_detach, to_np=False)
                dice_score_TC = cal_dice_score(pr[1], gt[1], need_detach, to_np=False)
                dice_score_WT = cal_dice_score(pr[2], gt[2], need_detach, to_np=False)

            else:
                dice_loss_ET = self.cal_dice_loss(y_pred[i][0], y_gt[i][0])
                dice_loss_TC = self.cal_dice_loss(y_pred[i][1], y_gt[i][1])
                dice_loss_WT = self.cal_dice_loss(y_pred[i][2], y_gt[i][2])
                dice_loss += (dice_loss_ET + dice_loss_TC + dice_loss_WT) / 3.0

                dice_score_ET = cal_dice_score(y_pred[i][0], y_gt[i][0], need_detach)
                dice_score_TC = cal_dice_score(y_pred[i][1], y_gt[i][1], need_detach)
                dice_score_WT = cal_dice_score(y_pred[i][2], y_gt[i][2], need_detach)

            dice_score['Dice_ET'] += dice_score_ET
            dice_score['Dice_TC'] += dice_score_TC
            dice_score['Dice_WT'] += dice_score_WT

        dice_score['Dice_ET'] /= y_gt.shape[0]
        dice_score['Dice_TC'] /= y_gt.shape[0]
        dice_score['Dice_WT'] /= y_gt.shape[0]
        dice_loss /= y_gt.shape[0]

        return dice_loss, dice_score


class GeneralizedDiceLoss(_Loss):
    def __init__(self, config):
        super(GeneralizedDiceLoss, self).__init__()
        self.config = config

    def cal_dice_loss(self, pred, gt):
        eps = self.config['dice_epsilon']
        if self.config['dice_square']:
            gt_pr = torch.sum(torch.mul(pred, gt))
            gt_gt = torch.sum(torch.mul(gt, gt))
            pr_pr = torch.sum(torch.mul(pred, pred))
            dice_eps = (2. * gt_pr + eps) / (gt_gt + pr_pr + eps)
        else:
            tp = torch.sum(torch.mul(pred, gt))
            fp = torch.sum(torch.mul(pred, 1 - gt))
            fn = torch.sum(torch.mul(1 - pred, gt))
            dice_eps = (2. * tp + eps) / (2 * tp + fp + fn + eps)
        return 1 - dice_eps

    def cal_gen_dice_loss(self, pred, gt):
        denominator = self.config['dice_epsilon']
        numerator = self.config['dice_epsilon']
        for j in range(pred.shape[0]):
            tp = torch.sum(torch.mul(pred[j], gt[j]))
            fp = torch.sum(torch.mul(pred[j], 1 - gt[j]))
            fn = torch.sum(torch.mul(1 - pred[j], gt[j]))
            w = 1 / ((torch.sum(gt[j])) ** 2 + self.config['dice_epsilon'])
            numerator += w * tp
            denominator += w * (2 * tp + fp + fn)
        return 1 - 2 * numerator / denominator

    def forward(self, y_pred, y_gt, need_detach=False, only_WT=False):
        dice_score = dict()
        dice_score['Dice_ET'] = 0.0
        dice_score['Dice_TC'] = 0.0
        dice_score['Dice_WT'] = 0.0
        dice_loss = 0.0

        for i in range(y_gt.shape[0]):
            if self.config['softmax']:
                dice_loss = self.cal_gen_dice_loss(y_pred[i].clone(), y_gt[i].clone())

                pr, gt = label_trans(y_pred[i].clone(), y_gt[i].clone(), need_detach)


                dice_score_ET = cal_dice_score(pr[0], gt[0], need_detach, to_np=False)
                dice_score_TC = cal_dice_score(pr[1], gt[1], need_detach, to_np=False)
                dice_score_WT = cal_dice_score(pr[2], gt[2], need_detach, to_np=False)

            else:
                dice_loss = self.cal_gen_dice_loss(y_pred[i], y_gt[i])

                dice_score_ET = cal_dice_score(y_pred[i][0], y_gt[i][0], need_detach)
                dice_score_TC = cal_dice_score(y_pred[i][1], y_gt[i][1], need_detach)
                dice_score_WT = cal_dice_score(y_pred[i][2], y_gt[i][2], need_detach)

            dice_score['Dice_ET'] += dice_score_ET
            dice_score['Dice_TC'] += dice_score_TC
            dice_score['Dice_WT'] += dice_score_WT

        dice_score['Dice_ET'] /= y_gt.shape[0]
        dice_score['Dice_TC'] /= y_gt.shape[0]
        dice_score['Dice_WT'] /= y_gt.shape[0]
        dice_loss /= y_gt.shape[0]

        return dice_loss, dice_score


class BinaryFocalLoss(_Loss):
    def __init__(self, config):
        super(BinaryFocalLoss, self).__init__()
        self.config = config
        self.gamma = config['focal_gamma']
        self.epsilon = 1e-8

    def forward(self, pr, gt):
        B, C, L, W, H = gt.size()
        N = L * W * H
        alpha = torch.sum(gt, dim=[2, 3, 4]) / N
        alpha = alpha.view(B, C, 1, 1, 1)
        alpha = alpha.expand(B, C, L, W, H)
        bfl = - alpha * gt * ((1 - pr) ** self.gamma) * torch.log(pr + self.epsilon) - (1 - alpha) * (1 - gt) * (
                    pr ** self.gamma) * torch.log(1 - pr + self.epsilon)
        return torch.sum(bfl)

class CrossEntropy(_Loss):
    def __init__(self, config):
        super(CrossEntropy, self).__init__()
        self.weight = torch.tensor([ 1.0000, 59.4507, 23.3190, 71.8481]).unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(config['device'])

    def forward(self, pr, gt, gamma=None):
        pr = pr.clone()
        pr[pr < 5e-3] = 5e-3
        if gamma is not None:
            ce_loss = - (gamma * torch.sum(gt.clone() * torch.log(pr.clone()) * self.weight, dim=1)).mean()
        else:
            ce_loss = - torch.sum(gt.clone() * torch.log(pr.clone()) * self.weight, dim=1).mean()
        return ce_loss


class TotalLoss(_Loss):
    def __init__(self, config):
        super(TotalLoss, self).__init__()
        self.config = config
        if self.config['focal_loss']:
            self.focal_loss = BinaryFocalLoss(config)

        if self.config['dice_loss']:
            if self.config['generalised_dice']:
                self.soft_dice_loss = GeneralizedDiceLoss(config)
            else:
                self.soft_dice_loss = SoftDiceLoss(config)

        if self.config['bce_loss']:
            self.bce_loss = nn.BCELoss(reduction='mean')

        if self.config['ce_loss']:
            self.ce_loss = CrossEntropy(config)


        if self.config['KL_loss']:
            self.KL_loss = KLLoss(config)

        if self.config['Att_loss']:
            self.Att_loss = ATTLoss(config)
        if self.config['contrastive_loss']:
            self.ct_loss = ContrastiveLoss(config)
    def update_ce_weight(self, device):
        self.ce_loss.weight *= 0.98
        self.ce_loss.weight = torch.max(self.ce_loss.weight, torch.ones(self.ce_loss.weight.shape).to(device))

    def forward(self, y_pred, y_gt, similarity=True, soft=None, de=None, features=None):
        loss_dict = dict()
        loss_dict['total_loss'] = 0

        # Segmentation loss
        if similarity:
            if self.config['dice_loss']:
                # print("source")
                loss_dict['dice_loss_source'], dice_source = self.soft_dice_loss(y_pred[0], y_gt)
                loss_dict['dice_loss_source'] *= self.config['dice_weight']

                # print("target")
                loss_dict['dice_loss_target'], dice_target = self.soft_dice_loss(y_pred[1], y_gt)
                loss_dict['dice_loss_target'] *= self.config['dice_weight']

                if self.config['train_mode'] == 'Mutually':
                    loss_dict['total_loss'] += loss_dict['dice_loss_source']
                loss_dict['total_loss'] += loss_dict['dice_loss_target']

            if self.config['bce_loss']:
                loss_dict['bce_loss_source'] = self.bce_loss(y_pred[0], y_gt)
                loss_dict['bce_loss_source'] *= self.config['bce_weight']

                loss_dict['bce_loss_target'] = self.bce_loss(y_pred[1], y_gt)
                loss_dict['bce_loss_target'] *= self.config['bce_weight']

                if self.config['train_mode'] == 'Mutually':
                    loss_dict['total_loss'] += loss_dict['bce_loss_source']
                loss_dict['total_loss'] += loss_dict['bce_loss_target']

            if self.config['ce_loss']:
                loss_dict['ce_loss_source'] = self.ce_loss(y_pred[0], y_gt)
                loss_dict['ce_loss_source'] *= self.config['ce_weight']

                loss_dict['ce_loss_target'] = self.ce_loss(y_pred[1], y_gt)
                loss_dict['ce_loss_target'] *= self.config['ce_weight']

                distance = loss_dict['ce_loss_target'] - loss_dict['ce_loss_source']
                distance[distance < 0] = 0.
                equal = torch.argmax(y_pred[0], dim=1) == torch.argmax(y_gt, dim=1)
                gamma = distance * equal
                gamma = gamma.detach()

                if self.config['train_mode'] == 'Mutually':
                    loss_dict['total_loss'] += loss_dict['ce_loss_source']
                loss_dict['total_loss'] += loss_dict['ce_loss_target']

            if self.config['focal_loss']:
                loss_dict['focal_loss_source'] = self.focal_loss(y_pred[0], y_gt)
                loss_dict['focal_loss_source'] *= self.config['focal_weight']

                loss_dict['focal_loss_target'] = self.focal_loss(y_pred[1], y_gt)
                loss_dict['focal_loss_target'] *= self.config['focal_weight']

                if self.config['train_mode'] == 'Mutually':
                    loss_dict['total_loss'] += loss_dict['focal_loss_source']
                loss_dict['total_loss'] += loss_dict['focal_loss_target']

            if self.config['kd_loss']:
                # print("kd")
                #loss_dict['kd_dice_loss'], _ = self.soft_dice_loss(soft[1], y_pred[0], need_detach=True)
                #loss_dict['kd_dice_loss'] *= self.config['kd_weight']
                #loss_dict['total_loss'] += loss_dict['kd_dice_loss']

                loss_dict['kd_ce_loss'] = self.ce_loss(soft[1], soft[0], gamma)
                loss_dict['kd_ce_loss'] *= self.config['kd_weight']
                loss_dict['total_loss'] += loss_dict['kd_ce_loss']

            if self.config['KL_loss']:
                f_source, f_target = features
                loss_dict['KL_loss'] = self.KL_loss(f_target, f_source)
                loss_dict['KL_loss'] *= self.config['KL_weight']
                loss_dict['total_loss'] += loss_dict['KL_loss']

            if self.config['Att_loss']:
                f_source, f_target = features
                loss_dict['Att_loss'] = self.Att_loss(f_target, f_source)
                loss_dict['Att_loss'] *= self.config['Att_weight']
                loss_dict['total_loss'] += loss_dict['Att_loss']

            if self.config['contrastive_loss']:
                f_source, f_target = features
                loss_dict['contrastive_loss'] = self.ct_loss(de[1], de[0])
                loss_dict['contrastive_loss'] *= self.config['contrastive_weight']
                loss_dict['total_loss'] += loss_dict['contrastive_loss']

            return loss_dict, [dice_source, dice_target]
        else:
            #y_pred_cat = torch.cat(y_pred, dim=0)  # to concatenate the results and treat them as batch
            #y_gts = [y_gt for _ in y_pred]  # to get a groundtruth with same batch size
            #y_gt_cat = torch.cat(y_gts, dim=0)

            if self.config['dice_loss']:
                loss_dict['dice_loss_target'], dice = self.soft_dice_loss(y_pred[0], y_gt)
                loss_dict['total_loss'] += loss_dict['dice_loss_target']


            if self.config['ce_loss']:
                loss_dict['ce_loss_target'] = self.ce_loss(y_pred[0], y_gt)
                loss_dict['ce_loss_target'] *= self.config['ce_weight']
                loss_dict['total_loss'] += loss_dict['ce_loss_target']

            return loss_dict, dice



def hausdorff(X, Y):
    return max(directed_hausdorff(X, Y)[0], directed_hausdorff(Y, X)[0])

def prec_sens_spe(X, Y):
    """return precion, sensitivity, and specificity"""
    TP = np.sum((X==1) & (Y==1))
    FN = np.sum((Y==1) & (X==0))

    TN = np.sum((X == 0) & (Y == 0))
    FP = np.sum((Y == 0) & (X == 1))
    return TP / (TP + FP + 1e-12), TP / (TP+FN+1e-12), TN / (TN + FP+1e-12)

class TestLoss(_Loss):
    def __init__(self, config):
        super(TestLoss, self).__init__()
        self.config = config
        if self.config['dice_loss']:
            self.soft_dice_loss = SoftDiceLoss(config)



    def forward(self, y_pred, y_gt, similarity=True, soft=None, de=None, features=None):
        loss_dict = dict()
        loss_dict['total_loss'] = 0

        # Segmentation loss
        if similarity:
            if self.config['dice_loss']:
                # print("source")
                loss_dict['dice_loss_source'], dice_source = self.soft_dice_loss(y_pred[0], y_gt)
                loss_dict['dice_loss_source'] *= self.config['dice_weight']

                # print("target")
                loss_dict['dice_loss_target'], dice_target = self.soft_dice_loss(y_pred[1], y_gt)
                loss_dict['dice_loss_target'] *= self.config['dice_weight']

            return loss_dict, [dice_source, dice_target]

        else:
            if self.config['dice_loss']:
                loss_dict['dice_loss_target'], dice = self.soft_dice_loss(y_pred[0], y_gt)
                loss_dict['total_loss'] += loss_dict['dice_loss_target']
                
                pred, gt = label_trans(y_pred[0][0], y_gt[0], need_detach=True)
                if pred[0].sum() < 50:
                    pred[0][pred[0] == 1.] = 0
                prec_et, sens_et, spe_et = prec_sens_spe(pred[0], gt[0])
                prec_tc, sens_tc, spe_tc = prec_sens_spe(pred[1], gt[1])
                prec_wt, sens_wt, spe_wt = prec_sens_spe(pred[2], gt[2])
                
                prec = {}
                prec['ET'] = prec_et
                prec['TC'] = prec_tc
                prec['WT'] = prec_wt

                sens = {}
                sens['ET'] = sens_et
                sens['TC'] = sens_tc
                sens['WT'] = sens_wt

                spe = {}
                spe['ET'] = spe_et
                spe['TC'] = spe_tc
                spe['WT'] = spe_wt
                
                hauss_et = hausdorff_distance(pred[0]==1, gt[0]==1)
                hauss_tc = hausdorff_distance(pred[1]==1, gt[1]==1)
                hauss_wt = hausdorff_distance(pred[2]==1, gt[2]==1)
                if np.isinf(hauss_et):
                    hauss_et = 100.
                if np.isinf(hauss_tc):
                    hauss_tc = 100.

                hauss = {}
                hauss['ET'] = hauss_et
                hauss['TC'] = hauss_tc
                hauss['WT'] = hauss_wt

            return loss_dict, dice, prec, sens, spe, hauss

