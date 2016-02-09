from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pickle
import os


def read_results(label, folderpath):
    filepath = os.path.join(folderpath, 'results_{}.pkl'.format(label))
    with open(filepath, 'rb') as f:
        res = pickle.load(f)
    return res


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


if __name__ == '__main__':

    from cnn_backwork import model_options

    num_examples = model_options['num_examples']

    proc_all_results(num_examples, 'misclass')
