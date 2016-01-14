import logging
import os
import pickle
from collections import OrderedDict

import numpy as np
import tensorflow as tf

import input_data

logger = logging.getLogger(__name__)


def load_params(fp, constant=False):

    with open(fp, 'rb') as f:
        od = pickle.load(f)

    constructor = tf.constant if constant else tf.Variable
    params = OrderedDict([(name, (constructor(val[0]), constructor(val[1]))) for name, val in od.items()])

    return params


def init_params(num_features, conv_layer_channels, num_fc_units):

    od = OrderedDict()

    for idx, (in_channels, out_channels) in enumerate(zip(conv_layer_channels[:-1], conv_layer_channels[1:])):

        W = weight_variable((FILTER_DIM_SIZE, FILTER_DIM_SIZE, in_channels, out_channels), name='W_conv{}'.format(idx))
        b = bias_variable((out_channels,), name='b_conv{}'.format(idx))
        od['conv{}'.format(idx)] = (W, b)

    W_fc = weight_variable((num_features * out_channels, num_fc_units), name='W_fc{}'.format(idx))
    b_fc = bias_variable((num_fc_units,), name='b_fc{}'.format(idx))
    od['fc'] = (W_fc, b_fc)

    W_softmax = weight_variable((num_fc_units, 10), name='W_softmax{}'.format(idx))
    b_softmax = bias_variable((10,), name='b_softmax{}'.format(idx))
    od['softmax'] = (W_softmax, b_softmax)

    return od


def save_params(params, fp):

    od = OrderedDict([(name, (tensor[0].eval(), tensor[1].eval())) for name, tensor in params.items()])

    with open(fp, 'wb') as f:
        pickle.dump(od, f)


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


def build_model(model_options, const_params=False):

    num_features = model_options['num_features']
    num_fc_units = model_options['num_fc_units']
    fp_params = model_options['fp_params']
    conv_layer_channels = model_options['conv_layer_channels']

    if os.path.isfile(fp_params):
        params = load_params(fp_params, constant=const_params)
    else:
        params = init_params(num_features, conv_layer_channels, num_fc_units)

    layers = []

    if const_params:
        x = tf.Variable(tf.random_uniform([model_options['image_dim_size']] * 2, 0, 1))
    else:
        x = tf.placeholder("float", shape=[None, IMAGE_DIM_SIZE ** 2])
    y = tf.placeholder("float", shape=[None, NUM_CLASSES])

    # Need to clip since might backwork values outside [0, 1]
    x = tf.clip_by_value(x, 0, 1)

    # These are the correctly formatted images
    image = tf.reshape(x, (-1, IMAGE_DIM_SIZE, IMAGE_DIM_SIZE, 1))
    layers.append(image)

    params_list = list(params.values())
    for idx, (W, b) in enumerate(params_list[:-2]):

        image = tf.nn.conv2d(image, W, strides=[1, 1, 1, 1], padding='SAME', name='conv{}_conv'.format(idx)) + b
        # image = tf.nn.sigmoid(image, name='conv{}_sigmoid'.format(idx))
        layers.append(image)

    image_flat = tf.reshape(image, (-1, num_features * conv_layer_channels[-1]))

    W_fc, b_fc = params_list[-2]
    fc_layer = tf.matmul(image_flat, W_fc) + b_fc
    layers.append(fc_layer)

    W_softmax, b_softmax = params_list[-1]
    softmax_layer = tf.nn.softmax(tf.matmul(fc_layer, W_softmax) + b_softmax)
    layers.append(softmax_layer)

    return params, x, y, layers


def train(data, model_options):

    params, x, y, layers = build_model(model_options)
    softmax_layer = layers[-1]

    xent = -tf.reduce_sum(y * softmax_layer)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(xent)

    correct = tf.equal(tf.argmax(y, 1), tf.argmax(softmax_layer, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    sess = tf.Session()

    with sess:
        try:
            tf.initialize_all_variables().run()

            for i in range(20000):
                batch = data.train.next_batch(50)

                if i % 100 == 0:
                    batch_accuracy = accuracy.eval(feed_dict={
                        x: batch[0], y: batch[1]
                    })
                    logger.info("step %d, training accuracy %g", i, batch_accuracy)

                train_step.run(feed_dict={x: batch[0], y: batch[1]})

            test_accuracy = accuracy.eval(feed_dict={x: data.test.images, y: data.test.labels})
            logger.info("test accuracy %g", test_accuracy)
        except KeyboardInterrupt:
            logger.info('Stopping training')
        finally:
            save_params(params, model_options['fp_params'])


def compute_layers_fixed_label(data, label, x, layers, sess):

    labels = data.train.labels.argmax(1)
    mask = labels == label

    data_label = data.train.images[mask]

    layers_label = sess.run(layers, feed_dict={x: data_label})

    return layers_label


def get_cost_from_layer(layer, layer_label):

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


def get_faulty_input_layer(data, model_options, cost_factors, label=2, num_opt_steps=1000):

    params, x, y, layers = build_model(model_options, const_params=True)

    with tf.Session() as sess:

        cost = build_cost(data, x, layers, cost_factors, label, sess)

        optimize_input_layer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

        tf.initialize_all_variables().run()
        import time
        t_start = time.time()

        try:
            for i in range(num_opt_steps):
                t0 = time.time()
                optimize_input_layer.run()
                logger.debug('%f seconds in optimization cycle with cost %f', time.time() - t0, cost.eval())
        except KeyboardInterrupt:
            pass
        logger.info('Average of %f seconds/optimization cycle over %f optimization cycles', (time.time() - t_start) / i, i)

        # layer_softmax_label = layers_label[-1]
        # print(tf.reduce_mean(layer_softmax_label, reduction_indices=0).eval())
        # print(x.eval())

        return x.eval()


def prop_forward(input_, model_options):

    params, x, y, layers = build_model(model_options)
    softmax_layer = layers[-1]

    input_ = input_.reshape(-1, model_options['num_features'])

    with tf.Session():
        tf.initialize_all_variables().run()
        res = softmax_layer.eval(feed_dict={x: input_})

    return res


if __name__ == '__main__':

    FP_PARAMS = 'params_1_1_1.pkl'

    logger.setLevel(logging.DEBUG)
    import sys
    FORMAT = '%(levelname)s: %(message)s'
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    IMAGE_DIM_SIZE = 28
    FILTER_DIM_SIZE = 5
    NUM_CLASSES = 10
    POOL_DIM_SIZE = 2

    num_out_channels = [1, 1, 1]
    num_layers = len(num_out_channels) + 2

    cost_factors = np.logspace(-num_layers + 1, 0, num=num_layers)

    model_options = {'image_dim_size': IMAGE_DIM_SIZE,
                     'num_features': IMAGE_DIM_SIZE ** 2,
                     'fp_params': FP_PARAMS,
                     'conv_layer_channels': num_out_channels,
                     'num_fc_units': 1024}

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # train(mnist, model_options)

    input_ = get_faulty_input_layer(mnist, model_options, cost_factors)

    softmax_input = prop_forward(input_, model_options)
    save = input('save? [default yes, n.* for no]')
    if not save or save[0] != 'n':
        np.save('input_', input_)
    print(softmax_input)
