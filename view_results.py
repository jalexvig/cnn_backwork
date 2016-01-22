from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pickle

def read_results(label):
    with open('results_{}.pkl'.format(label), 'rb') as f:
        res = pickle.load(f)
    return res


def proc_results_label(label, imgs, softmaxs, axs):

    for idx, (val, softmax, ax) in enumerate(zip(imgs, softmaxs, axs)):
        prob = round(softmax[label], 5)
        ax.set_title(prob)
        ax.set_xlabel(idx)
        ax.set_ylabel(label)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_adjustable('box-forced')
        ax.imshow(val)

def proc_all_results(num_examples, save=True, res=None):

    fig = plt.figure()
    axs = ImageGrid(fig, rect=111, nrows_ncols=(10, num_examples),
                    axes_pad=0.5, label_mode='L')

    for label in range(10):

        # TODO: don't hardcode timestep
        if res is None:
            imgs, softmaxs = read_results(label)[-1]
        else:
            imgs, softmaxs = res[label]

        axs_label = axs[label * num_examples: (label + 1) * num_examples]

        proc_results_label(label, imgs, softmaxs, axs_label)

    plt.tight_layout()
    if save:
        fig.savefig('label_{}_shape_{}_step_0'.format(label, num_examples))
    else:
        plt.show()

if __name__ == '__main__':

    from cnn_backwork import model_options

    num_examples = model_options['num_examples']

    # view_results(1)
    proc_all_results(num_examples, save=False)
