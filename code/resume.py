import os
import torch
from tools.log import add_log
from main import train


def resume(model_name, checkpoint_name):
    checkpoint_path = os.path.join('../', 'result', model_name, 'pkl', checkpoint_name)
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']
    init_fold = config['fold']

    add_log(config, 'Continue training...')

    if not config['parallel']:
        for fold in range(init_fold, config['num_fold']+1):
            config['fold'] = fold
            config = train(config, fold == init_fold, checkpoint)
    else:
        config = train(config, True, checkpoint)
    add_log(config, "All complete...")