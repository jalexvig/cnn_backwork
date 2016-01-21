import functools
import logging
import pickle

import numpy as np
import tensorflow as tf

import input_data
from my_mnist_backwork_input_pool import build_model, model_options

### LOGGER SETTINGS
logger = logging.getLogger(__name__)


def compute_layers_fixed_label(data, label, x, layers, sess):

    labels = data.train.labels.argmax(1)
    mask = labels == label

    data_label = data.train.images[mask]

    layers_label = sess.run(layers, feed_dict={x: data_label})

    return layers_label


def get_cost_from_layer(layer, layer_label, label, prev_coeff=1, sim_coeff=1):

    # layer has dimensionality (num_gen_examples, 28, 28, 1)
    # layer_label has dimensionality (num_training_examples_label, 28, 28, 1)

    # TODO: These dimensionality calculations are not correct b/c there are non convolutional layers

    # layer_shape = layer.get_shape().as_list()

    layer2 = tf.expand_dims(layer, 1)
    layer_label2 = tf.expand_dims(layer_label, 1)
    layer_name = layer.name.lower()

    prev_dist_term = tf.pow(layer - layer_label2, 2)
    similarity_dist_term = tf.pow(layer - layer2, 2)

    if 'softmax' in layer_name:
        # Add all probabilities of guessing the wrong answer
        cost = 1 - layer[:, label]
        cost = tf.reduce_mean(cost)

        # cost = tf.reduce_sum(layer, reduction_indices=[1]) - layer[:, label]
        # cost = tf.reduce_max(cost)

        return (cost,)

    elif 'inputs' in layer_name or 'conv' in layer_name:
        # (num_training_examples_label, num_gen_examples, 28, 28, 1)
        # Get the mean of the minimum distances of generated inputs to training examples of label
        prev_dist_term = tf.reduce_mean(prev_dist_term, reduction_indices=[2, 3, 4])

        # overcount = similarity_dist_term[slice(0, layer_shape[0]), slice(0, layer_shape[0]), :, :, :]

    elif 'fc' in layer_name:
        pass
        # overcount = similarity_dist_term[slice(0, layer_shape[0]), slice(0, layer_shape[0]), :]

    else:
        logger.warn('Unknown layer type %s', layer_name)
        raise ValueError

    prev_dist_term = tf.reduce_min(prev_dist_term, reduction_indices=[0])
    prev_dist_term = tf.reduce_mean(prev_dist_term)

    # TODO: fix calculation of overcount (2x above)
    # # Take mean over all relevant differences
    # similarity_dist_term = tf.reduce_sum(similarity_dist_term)
    # similarity_dist_term -= overcount
    # similarity_dist_term /= np.product(layer_shape) * (layer_shape[0] - 1)

    similarity_dist_term = tf.reduce_mean(similarity_dist_term)

    prev_cost = prev_coeff / prev_dist_term
    sim_cost = sim_coeff / similarity_dist_term

    return (prev_cost, sim_cost)


def build_cost(data, x, layers, sess, model_options):

    cost_factors = model_options['cost_factors']
    label = model_options['label']
    sim_coeff = model_options['sim_coeff']
    prev_coeff = model_options['prev_coeff']

    layers_label = compute_layers_fixed_label(data, label, x, layers, sess)
    layers_label = [tf.constant(a, name='layer{}_fixed_label'.format(idx)) for idx, a in enumerate(layers_label)]

    costs = []
    for layer, layer_label in zip(layers, layers_label):
        # TODO: get layer_shape
        cost_tup = get_cost_from_layer(layer, layer_label, label, sim_coeff=sim_coeff, prev_coeff=prev_coeff)
        costs.append(cost_tup)

    costs = [f * c for f, cost_tup in zip(cost_factors, costs) for c in cost_tup]

    return costs


def eval_cnn(x_pf, softmax_layer, label, num_features, input_):

    input_val = input_.eval()

    softmax = softmax_layer.eval(feed_dict={x_pf: input_val.reshape(-1, num_features)})

    return input_val, softmax


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

    # Note: convolution + activation + pooling is considered 1 layer
    params, x, y, layers = build_model(model_options, const_params=True)
    params_pf, x_pf, y_pf, layers_pf = build_model(model_options)
    softmax_layer_pf = layers_pf[-1]

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

    if not (max_num_steps < np.inf or (tolerance > 0 or prob_cap_mean < 1 or prob_cap_min < 1) and min_num_steps < np.inf):
        logger.warn('no exit conditions for backworking CNN inputs')

    eval_cnn_wrapper = functools.partial(eval_cnn, x_pf, softmax_layer_pf, label, num_features)

    l = []

    with tf.Session() as sess:

        logger.info('building model')
        costs = build_cost(data, x, layers, sess, model_options)
        cost = sum(costs)

        # TODO: Add ability to only backprop through some examples in batch (don't need to adjust ones with good fit)
        optimize_input_layer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
        logger.info('starting backworking')

        tf.initialize_all_variables().run()
        import time
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


if __name__ == '__main__':

    import datetime
    FORMAT = '%(levelname)s: %(message)s'
    FILEPATH = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    FILEPATH += '_backwork_cnn.log'

    logging.basicConfig(
        level=logging.DEBUG,
        format=FORMAT,
        filename=FILEPATH,
    )

    num_layers = len(model_options['conv_layers']) + 3
    cost_factors = np.logspace(-num_layers, -1, num=num_layers)
    cost_factors[-1] *= 100

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    model_options_update = {
        'cost_factors': cost_factors,
        'save_freq': 1,
        'prob_cap_min': 0.999,
        'num_examples': 8,
        'prev_coeff': 0.5,
        'sim_coeff': 0.5,
        'num_mono_dec_saves': 5,
    }

    model_options.update(model_options_update)

    for label in range(10):

        model_options['label'] = label

        results = get_faulty_input_layer(mnist, model_options)
        with open('results_{}.pkl'.format(label), 'wb') as f:
            pickle.dump(results, f)
