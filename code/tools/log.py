import os
import time
import numpy as np


def create_log(config):
    result_dir = os.path.join(config['workspace_root'], "result")
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    result_path = os.path.join(result_dir, config['model_name'])
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    config['result_path'] = result_path
    add_log(config, "The log for " + result_path + " is created.")
    add_log(config, "\n==> Configurations:")

    for k, v in config.items():
        add_log(config, '[' + k + ']:\t' + str(v))

    add_log(config, "")

    return result_path


def add_log(config, msg, PRINT=True):
    log_path = os.path.join(config['result_path'], "log.txt")
    with open(log_path, "a") as f:
        f.write(time.strftime("%y%m%d %H:%M:%S\t", time.localtime()))
        f.write(msg)
        f.write("\n")
    if PRINT:
        print(msg)


def dict_to_msg(results, title, format='%.04f\t'):
    np.set_printoptions(precision=4, suppress=True)
    msg = '[%s]\t' % title
    for k, v in results.items():
        msg += k + ': ' + format % v
    return msg