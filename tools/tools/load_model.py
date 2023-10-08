from tools.log import add_log
from models.HUnet import UNet, HUNetv4
from models.PMKL_model import Teacher, KDNet
from collections import OrderedDict
import torch
import os


def load_teacher(config, model, from_UNet=True, load_student=False, for_HUNetv4=False):
    if from_UNet:
        checkpoint_name = 'Checkpoint_best_fold_%d.pkl' % config['fold']
        checkpoint_path = os.path.join('../', 'result', config['pretrained_teacher'], 'pkl', checkpoint_name)
        add_log(config, "Loading the pretrained teacher model from: " + checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        teacher_config = checkpoint['config']
        pretrained_model = UNet(teacher_config).to(config['device'])
        pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        pretrained_dict = pretrained_model.state_dict()
        new_state_dict = OrderedDict()
        for key, value in pretrained_dict.items():
            key_split = key.split('.')
            if key_split[0] == 'encoder_source':
                new_key = key
            elif key_split[0] == 'bottle_neck_source':
                new_key = key
            elif key_split[0] == 'decoder':
                new_key = 'decoder_source.' + '.'.join(key_split[1:])
            new_state_dict[new_key] = value

        model_dict = model.state_dict()
        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)

    elif load_student:
        checkpoint_name = 'Checkpoint_best_fold_%d.pkl' % config['fold']
        checkpoint_path = os.path.join('../', 'result', config['pretrained_student'], 'pkl', checkpoint_name)
        add_log(config, "Loading the pretrained teacher model from: " + checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        checkpoint_name = 'Checkpoint_best_fold_%d.pkl' % config['fold']
        checkpoint_path = os.path.join('../', 'result', config['pretrained_teacher'], 'pkl', checkpoint_name)
        add_log(config, "Loading the pretrained teacher model from: " + checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint

def load_teacher_PMKL(config, model):
    checkpoint_name = 'Checkpoint_best_fold_%d.pkl' % config['fold']
    checkpoint_path = os.path.join('../', 'result', config['pretrained_teacher'], 'pkl', checkpoint_name)
    add_log(config, "Loading the pretrained teacher model from: " + checkpoint_path)
    checkpoint = torch.load(checkpoint_path)

    model_dict = model.state_dict()
    model_dict.update(checkpoint["model_state_dict"])
    model.load_state_dict(model_dict)
    return model, checkpoint


def load_model_(config, model, reload=True):
    checkpoint_name =  'Checkpoint_best_fold_%d.pkl' % config['fold']
    checkpoint_path = os.path.join(config['result_path'], 'pkl', checkpoint_name)
    add_log(config, "Loading the pretrained teacher model from: " + checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    teacher_config = checkpoint['config']
    if teacher_config['Model'] == 'Baseline' and reload:
        pretrained_model = UNet(teacher_config).to(config['device'])
        pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        pretrained_dict = pretrained_model.state_dict()
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint

