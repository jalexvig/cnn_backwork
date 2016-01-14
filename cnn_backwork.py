import logging
import pickle
import sys

import numpy as np
import tensorflow as tf

import input_data
from my_mnist_backwork_input_pool import build_model, model_options

### LOGGER SETTINGS
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
FORMAT = '%(levelname)s: %(message)s'
handler = logging.StreamHandler(sys.stderr)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


def compute_layers_fixed_label(data, label, x, layers, sess):

    labels = data.train.labels.argmax(1)
    mask = labels == label

    data_label = data.train.images[mask]

    layers_label = sess.run(layers, feed_dict={x: data_label})

    return layers_label


def get_cost_from_layer(layer, layer_label):

    # layer has dimensionality (num_gen_inputs, 28, 28, 1)
    # layer_label has dimensionality (num_training_examples_label, 28, 28, 1)

    # This will compute the mean over all randomly generated inputs as well as layer evaluated over all training data
    layer = tf.expand_dims(layer, 1)
    cost = tf.reduce_mean(tf.pow(layer - layer_label, 2))

    return cost


def build_cost(data, x, layers, cost_factors, label, sess):

    layers_label = compute_layers_fixed_label(data, label, x, layers, sess)
    layers_label = [tf.constant(a, name='layer{}_fixed_label'.format(idx)) for idx, a in enumerate(layers_label)]

    costs = [get_cost_from_layer(layer, layer_label) for layer, layer_label in zip(layers[:-1], layers_label[:-1])]

    # TODO(jalex): figure out if there is a better way of doing this
    cost_softmax_layer = tf.reduce_sum(layers[-1]) - tf.reduce_sum(layers[-1][:, label])

    # Seems like this isn't needed
    # TODO(jalex): figure out if there is a better way of doing this (than adding a const)
    # cost_softmax_layer += 1 / (layers[-1][:, label] + 1e-10)

    costs.append(cost_softmax_layer)

    cost = sum(f * c for f, c in zip(cost_factors, costs))

    return cost


def get_faulty_input_layer(data, model_options):

    # Note: convolution + activation + pooling is considered 1 layer
    params, x, y, layers = build_model(model_options, const_params=True)

    save_freq = model_options.get('save_freq', 20)
    cost_factors = model_options['cost_factors']
    label = model_options['label']
    min_num_steps = model_options.get('min_num_steps', 0)
    max_num_steps = model_options.get('max_num_steps', np.inf)
    tolerance = model_options.get('tolerance', 0)
    prob_cap_mean = model_options.get('prob_cap_mean', 1)
    prob_cap_min = model_options.get('prob_cap_min', 1)

    l = []

    with tf.Session() as sess:

        cost = build_cost(data, x, layers, cost_factors, label, sess)

        optimize_input_layer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

        tf.initialize_all_variables().run()
        import time
        t_start = time.time()

        cost_val_old = np.inf
        i = 0

        try:
            while True:

                if i >= max_num_steps:
                    break

                t0 = time.time()
                optimize_input_layer.run()
                cost_val_new = cost.eval()
                logger.debug('%f seconds in optimization cycle with cost %f', time.time() - t0, cost_val_new)
                if (save_freq and i % save_freq == 0) or i == max_num_steps - 1:
                    val = x.eval()
                    softmax = prop_forward(val, model_options)
                    l.append((val, softmax))

                    prob_correct_mean = softmax[:, label].mean()
                    prob_correct_min = softmax[:, label].min()
                    logger.debug('%f / %f min / mean probability of correct label', prob_correct_min, prob_correct_mean)
                    if (prob_correct_mean > prob_cap_mean or prob_correct_min > prob_cap_min) and i >= min_num_steps:
                        break

                if abs(cost_val_new - cost_val_old) < tolerance and i >= min_num_steps:
                    break

                i += 1

                cost_val_old = cost_val_new
        except KeyboardInterrupt:
            pass
        logger.info('Average of %f seconds/optimization cycle over %f optimization cycles', (time.time() - t_start) / i, i)

        return l


def prop_forward(input_, model_options):

    fp_params = model_options['fp_params']

    params, x, y, layers = build_model(model_options, fp_params)
    softmax_layer = layers[-1]

    input_ = input_.reshape(-1, model_options['num_features'])

    with tf.Session():
        tf.initialize_all_variables().run()
        res = softmax_layer.eval(feed_dict={x: input_})

    return res

if __name__ == '__main__':

    num_layers = len(model_options['conv_layers']) + 2
    cost_factors = np.logspace(-num_layers + 1, 0, num=num_layers)

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    model_options_update = {
        'cost_factors': cost_factors,
        'save_freq': 4,
        'prob_cap_min': 0.99,
        'num_examples': 4,
    }

    # TODO: fix num_examples

    model_options.update(model_options_update)

    for label in range(10):

        model_options['label'] = label

        results = get_faulty_input_layer(mnist, model_options)
        with open('results_{}.pkl'.format(label), 'wb') as f:
            pickle.dump(results, f)
