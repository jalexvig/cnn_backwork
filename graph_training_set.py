import numpy as np
import tensorflow as tf

import input_data
from cnn_backwork import build_model
from view_results import proc_all_results


def get_permuted_train_data(data):

    l = []

    labels = data.train.labels.argmax(1)

    for label in range(10):
        mask = labels == label
        data_label = data.train.images[mask]
        l.append(np.random.permutation(data_label))

    return l


def prop_forward(data, model_options, use_worst):

    params_pf, x_pf, y_pf, layers_pf = build_model(model_options)
    softmax_layer_pf = layers_pf[-1]

    num_keep = model_options['num_examples']
    if not use_worst:
        data = [d[:num_keep] for d in data]

    new_data = []

    with tf.Session():

        tf.initialize_all_variables().run()

        res = []

        for label, data_label in enumerate(data):

            probs = softmax_layer_pf.eval(feed_dict={x_pf: data_label})
            if use_worst:
                order = np.argsort(probs[:, label])[:num_keep]
                probs = probs[order]
                data_label = data_label[order]

            new_data.append(data_label)
            res.append(probs)

    return res, new_data


def graph_subset_train_data(data, model_options, save=False, use_worst=False):

    image_dim_size = model_options['image_dim_size']

    data_subset = get_permuted_train_data(data)

    probs, data_subset = prop_forward(data_subset, model_options, use_worst)

    data_subset = [d.reshape(-1, image_dim_size, image_dim_size) for d in data_subset]

    res = list(zip(data_subset, probs))

    fname = model_options.get('fp_save', None)

    proc_all_results(model_options['num_examples'], save_filename=fname, res=res)


if __name__ == '__main__':

    from cnn_backwork import model_options

    model_options['num_examples'] = 8
    model_options['fp_params'] = 'params_norm3.pkl'

    model_options['fp_save'] = 'worst_train_data'

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    graph_subset_train_data(mnist, model_options, save=True, use_worst=True)
