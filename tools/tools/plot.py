import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
matplotlib.use('Agg')


def append_history(hist_1, hist_2=None):
    hist = dict()
    if hist_2 is not None:
        for (k, v) in hist_1.items():
            hist[k] = hist_1[k] + hist_2[k]
    else:
        hist = hist_1
    return hist


def plot_history(config, history, fig_name='', key_select=None):
    colors = ['y-', 'b-', 'g-', 'r-', 'y--', 'b--', 'g--', 'r--']
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    i = 0

    if key_select is None:
        for (k, v) in history.items():
            epoch_list = list(range(len(v)))
            plt.plot(epoch_list, v, colors[i], label="%s" % k)
            i += 1

    plt.title(fig_name)
    plt.xlabel("Iters")
    plt.ylabel("History")
    plt.legend()

    fig_name = fig_name + 'fold_%d.png' % config['fold']
    plot_dir = os.path.join(config['result_path'], "loss")
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    fig_path = os.path.join(plot_dir, fig_name)
    plt.savefig(fig_path)
    plt.close()


# def plot_history(config, histories, fig_name='', key_select=None):
#
#     colors = ['y-', 'b-', 'g-', 'r-', 'y--', 'b--', 'g--', 'r--']
#     fig = plt.figure(figsize=(7, 5))
#     ax1 = fig.add_subplot(1, 1, 1)
#     i = 0
#     train_history, valid_history = histories
#     if key_select is None:
#         for (k, v) in train_history.items():
#             epoch_list = list(range(len(v)))
#             plt.plot(epoch_list, v, colors[i], label="train_%s" % k)
#             if k in valid_history.keys():
#                 plt.plot(epoch_list, valid_history[k], colors[i+4], label="valid_%s" % k)
#             i += 1
#     else:
#         for k in key_select:
#             epoch_list = list(range(len(train_history[k])))
#             plt.plot(epoch_list, train_history[k], colors[i], label="train_%s" % k)
#             if k in valid_history.keys():
#                 plt.plot(epoch_list, valid_history[k], colors[i + 4], label="valid_%s" % k)
#         fig_name = fig_name + '_'.join(key_select)
#
#     plt.title(fig_name)
#     plt.xlabel("Iters")
#     plt.ylabel("History")
#     plt.legend()
#
#     fig_name = fig_name + 'fold_%d.png' % config['fold']
#     plot_dir = os.path.join(config['result_path'], "loss")
#     if not os.path.exists(plot_dir):
#         os.mkdir(plot_dir)
#     fig_path = os.path.join(plot_dir, fig_name)
#     plt.savefig(fig_path)
#     plt.close()


def plot_nib(img, name, config, channel_name):
    fig_dir = os.path.join(config['result_path'], 'fig')
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    nii_dir = os.path.join(fig_dir, 'nii')
    if not os.path.exists(nii_dir):
        os.mkdir(nii_dir)

    print(img.shape)
    img = img.cpu().detach().numpy()
    img = np.squeeze(img)

    for c in range(img.shape[0]):
        nib_data = img[c]
        print('nib_data:', nib_data.shape)
        nii_name = name + channel_name[c] + ".nii.gz"
        nii_path = os.path.join(nii_dir, nii_name)
        nib_data = nib.Nifti1Image(nib_data, np.eye(4))
        nib.save(nib_data, nii_path)
