import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from salient_features import get_grads, get_salient_features
from utils import read_results


def proc_results_label(label, imgs, softmaxs, axs):

    for idx, (val, softmax, ax) in enumerate(zip(imgs, softmaxs, axs)):
        prob = round(softmax[label], 5)
        ax.set_title(prob)
        # ax.set_xlabel(idx)
        ax.set_ylabel(label)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_adjustable('box-forced')
        ax.imshow(val)


def proc_all_results(num_examples, save_filename=None, res=None, folderpath=''):

    # TODO: modify this to use graph_grid

    num_results = 10

    fig = plt.figure(figsize=(12, 12))
    axs = ImageGrid(fig, rect=111, nrows_ncols=(num_results, num_examples),
                    axes_pad=0.35, label_mode='L')

    for label in range(num_results):

        # TODO: don't hardcode timestep
        if res is None:
            imgs, softmaxs = read_results(label, folderpath)[-1]
        else:
            imgs, softmaxs = res[label]

        axs_label = axs[label * num_examples: (label + 1) * num_examples]

        proc_results_label(label, imgs, softmaxs, axs_label)

    plt.tight_layout()
    if save_filename:
        fig.savefig(save_filename, bbox_inches='tight')
    else:
        plt.show()


def graph_grid(imgs, shape=None):

    if shape is not None:
        imgs = imgs.reshape(shape)

    fig = plt.figure(figsize=(12, 12))
    axs = ImageGrid(fig, rect=111, nrows_ncols=imgs.shape[:2],
                    axes_pad=0.35, label_mode='L')

    for i in range(imgs.shape[0]):

        axs_slice = slice(i * imgs.shape[1], (i + 1) * imgs.shape[1])

        for img, ax in zip(imgs[i], axs[axs_slice]):

            ax.set_ylabel(i)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(img)

    plt.tight_layout()

    plt.show()


def view_all_salient_features(model_options, labels=None, num_examples=None, folderpath='', data=None):

    imgs = []

    for label in range(10):

        if data is None:
            imgs_label, _ = read_results(label, folderpath)[-1]
        else:
            imgs_label = data[label]

        imgs_label = imgs_label[0:1, :, :]
        salient_features = get_salient_features(model_options, imgs_label, label,
                                                grad_sign='pos', num_keep=200, as_binary=True, mul_weights=False)

        imgs.append(salient_features)

    imgs = np.array(imgs)

    if labels:
        imgs = imgs[labels]
    if num_examples:
        imgs = imgs[:, :num_examples]

    shape = list(imgs.shape[:2])
    shape += [model_options['image_dim_size']] * 2

    graph_grid(imgs, shape=shape)


if __name__ == '__main__':

    from cnn_backwork import model_options
    model_options['fp_params'] = 'params/params_norm4.pkl'

    # num_examples = model_options['num_examples']
    # proc_all_results(num_examples, 'misclass')

    # view_all_salient_features(model_options, folderpath='results/results_8_batch_min_prob_0999')
    view_all_salient_features(model_options, folderpath='results/reference_images', num_examples=3, labels=[2, 3, 4])
