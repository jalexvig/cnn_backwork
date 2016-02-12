import functools
import logging
import pickle
import time

import numpy as np
import tensorflow as tf

import input_data
from proc_mnist import build_model, model_options
from utils import get_data_label

### LOGGER SETTINGS
logger = logging.getLogger(__name__)


def compute_layers_fixed_label(data, label, x, layers, sess):

    data_label = get_data_label(data, label)

    layers_label = sess.run(layers, feed_dict={x: data_label})

    return layers_label


def get_cost_from_layer(layer, layer_label, label):

    # For convolutional layers, layer has dimensionality (num_gen_examples, 28, 28, 1) and
    # layer_label has dimensionality (num_training_examples_label, 28, 28, 1)

    layer_name = layer.name.lower()

    # reference_dist_tensor is the tensor that represents the distance from generated images to the reference images
    # batch_dist_tensor is the tensor that represents the distance from generated images to other images in the batch
    reference_dist_tensor = tf.pow(layer - tf.expand_dims(layer_label, 1), 2)
    batch_dist_tensor = tf.pow(layer - tf.expand_dims(layer, 1), 2)

    if 'softmax' in layer_name:
        # Average all probabilities of guessing the wrong answer
        cost = 1 - layer[:, label]
        cost = tf.reduce_mean(cost)

        return cost,

    elif 'inputs' in layer_name or 'conv' in layer_name:
        # reference_dist_tensor has dimensionality (num_training_examples_label, num_gen_examples, 28, 28, 1)
        # Get the mean of the minimum distances of generated inputs to training examples of label
        reference_dist_tensor = tf.reduce_mean(reference_dist_tensor, reduction_indices=[2, 3, 4])

    elif 'fc' in layer_name:
        pass

    else:
        raise ValueError('Unknown layer type on layer %s', layer_name)

    reference_dist_tensor = tf.reduce_min(reference_dist_tensor, reduction_indices=[0])
    reference_dist_tensor = tf.reduce_mean(reference_dist_tensor)

    # TODO: this compares a vector to itself (remove self comparisons)
    batch_dist_tensor = tf.reduce_mean(batch_dist_tensor)

    reference_cost = 1 / reference_dist_tensor
    batch_cost = 1 / batch_dist_tensor

    return reference_cost, batch_cost


def build_cost(data, x, layers, sess, model_options):

    layer_cost_coeffs = model_options['layer_cost_coeffs']
    label = model_options['label']
    batch_cost_coeff = model_options['batch_cost_coeff']
    reference_cost_coeff = model_options['reference_cost_coeff']

    layers_label = compute_layers_fixed_label(data, label, x, layers, sess)
    layers_label = [tf.constant(a, name='layer{}_fixed_label'.format(idx)) for idx, a in enumerate(layers_label)]

    costs = [get_cost_from_layer(layer, layer_label, label) for layer, layer_label in zip(layers, layers_label)]
    costs = [c for cost_tuple in costs for c in cost_tuple]

    weights = np.outer(layer_cost_coeffs[: -1], [reference_cost_coeff, batch_cost_coeff]).flatten()
    weights = np.append(weights, layer_cost_coeffs[-1])

    costs = (costs * weights)

    return costs


def eval_cnn(x_pf, softmax_layer, num_features, input_):

    input_val = input_.eval()

    softmax_val = softmax_layer.eval(feed_dict={x_pf: input_val.reshape(-1, num_features)})

    return input_val, softmax_val


class ProbTracker:

    def __init__(self, timesteps, batch_size):

        self._tracker = np.empty((timesteps, batch_size))
        self._tracker.fill(np.nan)

    def __call__(self, probs):

        for j, prob in enumerate(probs):
            nan_idxs = np.where(np.isnan(self._tracker[:, j]))[0]
            if len(nan_idxs):
                self._tracker[nan_idxs[0], j] = prob
            else:
                self._tracker[:-1, j] = self._tracker[1:, j]
                self._tracker[-1, j] = prob

    def get_mask(self):

        m = ~np.isnan(self._tracker).any(axis=0)
        m &= (np.diff(self._tracker, axis=0) <= 0).all(axis=0)
        # TODO: softcode this cap
        m &= self._tracker[-1] < 0.99

        return m

    def mark_replaced_inputs(self, replacement_mask):

        self._tracker[:, replacement_mask] = np.nan

    @property
    def values(self):

        return self._tracker


def get_faulty_input_layer(data, model_options):

    save_freq = model_options.get('save_freq', 20)
    label = model_options['label']
    num_features = model_options['num_features']
    min_num_steps = model_options.get('min_num_steps', 0)
    max_num_steps = model_options.get('max_num_steps', np.inf)
    tolerance = model_options.get('tolerance', 0)
    prob_cap_mean = model_options.get('prob_cap_mean', 1)
    prob_cap_min = model_options.get('prob_cap_min', 1)
    num_mono_dec_saves = model_options.get('num_mono_dec_saves', np.inf)
    batch_size = model_options.get('num_examples', 1)
    init_w_train_vals = model_options.get('init_w_train_vals', False)

    x_vals = None
    if init_w_train_vals:
        train_examples = get_data_label(data, label)
        x_vals = np.random.permutation(train_examples)
        x_vals = np.array_split(x_vals, 8)
        x_vals = np.array([np.average(xv, axis=0) for xv in x_vals])

    # Note: convolution + activation + pooling is considered 1 layer
    params, x, y, layers = build_model(model_options, const_params=True, x_values=x_vals)
    params_pf, x_pf, y_pf, layers_pf = build_model(model_options)
    softmax_layer_pf = layers_pf[-1]

    if not (max_num_steps < np.inf or (tolerance > 0 or prob_cap_mean < 1 or prob_cap_min < 1) and min_num_steps < np.inf):
        logger.warn('no exit conditions for backworking CNN inputs')

    eval_cnn_wrapper = functools.partial(eval_cnn, x_pf, softmax_layer_pf, num_features)

    l = []

    with tf.Session() as sess:

        logger.info('building model')
        costs = build_cost(data, x, layers, sess, model_options)
        cost = sum(costs)

        optimize_input_layer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
        logger.info('starting backworking')

        tf.initialize_all_variables().run()
        t_start = time.time()

        cost_val_old = np.inf
        i = 0

        probs_correct_tracker = ProbTracker(num_mono_dec_saves, batch_size)

        try:
            while True:

                if i >= max_num_steps:
                    break

                t0 = time.time()
                optimize_input_layer.run()
                cost_val_new = cost.eval()
                logger.info('%f seconds in optimization cycle %i with cost %f', time.time() - t0, i, cost_val_new)

                for idx, c in enumerate(costs):
                    layer = layers[idx // 2]
                    logger.debug('layer %s cost %f', layer.name, c.eval())

                if (save_freq and i % save_freq == 0) or i == max_num_steps - 1:
                    input_vals, softmax = eval_cnn_wrapper(x)

                    probs_correct = softmax[:, label]
                    probs_correct_tracker(probs_correct)

                    l.append((input_vals, softmax))

                    prob_correct_mean, prob_correct_min = probs_correct.mean(), probs_correct.min()
                    logger.info('%f / %f min / mean probability of correct label', prob_correct_min, prob_correct_mean)

                    if i >= min_num_steps and (prob_correct_mean > prob_cap_mean or prob_correct_min > prob_cap_min):
                        break

                    m = probs_correct_tracker.get_mask()

                    if m.any():
                        logger.info('reassignment mask %s', m)
                        logger.debug('probability history of %i steps: %s',
                                     num_mono_dec_saves, probs_correct_tracker.values[:, m])
                        shape = [m.sum()] + [model_options['image_dim_size']] * 2
                        input_vals[m] = np.random.rand(*shape)
                        probs_correct_tracker.mark_replaced_inputs(m)
                        sess.run(x.assign(input_vals))

                if abs(cost_val_new - cost_val_old) < tolerance and i >= min_num_steps:
                    break

                i += 1

                cost_val_old = cost_val_new
        except KeyboardInterrupt:
            pass
        logger.info('Average of %f seconds/optimization cycle over %f optimization cycles', (time.time() - t_start) / i, i)

        return l

num_layers = len(model_options['conv_layers']) + 3
layer_cost_coeffs = np.logspace(-num_layers, -1, num=num_layers)
layer_cost_coeffs[-1] *= 100

model_options_update = {
        'layer_cost_coeffs': layer_cost_coeffs,
        'save_freq': 1,
        'prob_cap_min': 0.999,
        'num_examples': 8,
        'reference_cost_coeff': 0.5,
        'batch_cost_coeff': 0.5,
        'num_mono_dec_saves': 3,
        'max_num_steps': 60,
    }

model_options.update(model_options_update)

if __name__ == '__main__':

    modifier = '_aug_misclass'
    model_options['fp_params'] = 'params/params_aug_misclass2.pkl'

    import datetime
    import os

    FORMAT = '%(levelname)s: %(message)s'
    FILEPATH = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    FILEPATH += '_backwork_cnn{}.log'.format(modifier)
    FILEPATH = os.path.join('logs', FILEPATH)

    logging.basicConfig(
        level=logging.DEBUG,
        format=FORMAT,
        filename=FILEPATH,
    )

    logger.info(model_options)

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    for label in range(10):

        model_options['label'] = label

        results = get_faulty_input_layer(mnist, model_options)
        with open('results{}_{}.pkl'.format(modifier, label), 'wb') as f:
            pickle.dump(results, f)
