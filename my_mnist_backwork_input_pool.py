import logging
import math
import os
import pickle
from collections import OrderedDict

import tensorflow as tf

import input_data

logger = logging.getLogger(__name__)


def _load_params(model_options, constant=False):

    fp_params = model_options['fp_params']

    if os.path.isfile(fp_params):
        with open(fp_params, 'rb') as f:
            od = pickle.load(f)
            num_fc_in = list(od.values())[-2][0].shape[0]
            constructor = tf.constant if constant else tf.Variable
            params = OrderedDict([(name, (constructor(val[0]), constructor(val[1]))) for name, val in od.items()])
    elif not constant:
        params, num_fc_in = _init_params(model_options)
    else:
        # TODO: init_params will always return params as variables - for completeness make able to return constant
        raise ValueError('Can\'t initialize constant params')

    return params, num_fc_in


def _init_params(model_options):

    conv_layers = model_options['conv_layers']
    pool_layers = model_options['pool_layers']

    image_dim_size_width = image_dim_size_height = model_options['image_dim_size']

    od = OrderedDict()

    for i, (conv_layer, pool_layer) in enumerate(zip(conv_layers, pool_layers)):

        filter_conv = conv_layer['filter']
        strides_conv = conv_layer['strides']
        padding_conv = conv_layer['padding']

        W = weight_variable(filter_conv, name='W_conv{}'.format(i))
        b = bias_variable(filter_conv[-1:], name='b_conv{}'.format(i))
        od['conv{}'.format(i)] = (W, b)

        if padding_conv == 'SAME':
            image_dim_size_height = math.ceil(image_dim_size_height / strides_conv[1])
            image_dim_size_width = math.ceil(image_dim_size_width / strides_conv[2])
        elif padding_conv == 'VALID':
            image_dim_size_height = (image_dim_size_height - filter_conv[0] + strides_conv[1]) // strides_conv[1]
            image_dim_size_width = (image_dim_size_width - filter_conv[1] + strides_conv[2]) // strides_conv[2]

        filter_pool = pool_layer['filter']
        strides_pool = pool_layer['strides']
        padding_pool = pool_layer['padding']

        if padding_pool == 'SAME':
            image_dim_size_height = math.ceil(image_dim_size_height / strides_pool[1])
            image_dim_size_width = math.ceil(image_dim_size_width / strides_pool[2])
        elif padding_pool == 'VALID':
            image_dim_size_height = (image_dim_size_height - filter_pool[0] + strides_pool[1]) // strides_pool[1]
            image_dim_size_width = (image_dim_size_width - filter_pool[1] + strides_pool[2]) // strides_pool[2]

    # TODO: this does not account for strided channels in either convolution or pooling
    num_fc_in = image_dim_size_height * image_dim_size_width * filter_conv[-1]
    num_fc_out = model_options['num_fc_units']
    od['fc'] = weight_variable((num_fc_in, num_fc_out)), bias_variable((num_fc_out,))

    num_classes = model_options['num_classes']
    od['softmax'] = weight_variable((num_fc_out, num_classes)), bias_variable((10,))

    return od, num_fc_in


def _save_params(params, fp):

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

    conv_layers = model_options['conv_layers']
    pool_layers = model_options['pool_layers']
    image_dim_size = model_options['image_dim_size']
    num_classes = model_options['num_classes']

    params, num_fc_in = _load_params(model_options, constant=const_params)

    layers = []

    if const_params:
        num_examples = model_options.get('num_examples', 1)
        shape = [num_examples] + [model_options['image_dim_size']] * 2
        # Need to clip since might backwork values outside [0, 1]
        x = tf.Variable(tf.clip_by_value(tf.random_uniform(shape, 0, 1), 0, 1))
    else:
        x = tf.placeholder("float", shape=[None, image_dim_size ** 2])
    y = tf.placeholder("float", shape=[None, num_classes])

    # These are the correctly formatted images
    image = tf.reshape(x, (-1, image_dim_size, image_dim_size, 1), name='inputs')
    layers.append(image)

    params_list = list(params.values())
    for i, (conv_layer, pool_layer, conv_params) in enumerate(zip(conv_layers, pool_layers, params_list[:-2])):

        W, b = conv_params

        image = tf.nn.conv2d(image, W, conv_layer['strides'], conv_layer['padding'], name='conv{}'.format(i))
        image += b
        image = conv_layer['act_fn'](image)
        image = tf.nn.max_pool(image, pool_layer['filter'], pool_layer['strides'], pool_layer['padding'], name='conv{}'.format(i))
        # image = tf.nn.sigmoid(image, name='conv{}_sigmoid'.format(idx))
        layers.append(image)

    image_flat = tf.reshape(image, (-1, num_fc_in))

    W_fc, b_fc = params_list[-2]
    fc_layer = tf.nn.xw_plus_b(image_flat, W_fc, b_fc, name='fc')
    layers.append(fc_layer)
    fc_layer = tf.nn.relu(fc_layer)

    W_softmax, b_softmax = params_list[-1]
    softmax_layer = tf.nn.softmax(tf.matmul(fc_layer, W_softmax) + b_softmax, 'softmax')
    layers.append(softmax_layer)

    return params, x, y, layers


def train(data, model_options):

    params, x, y, layers = build_model(model_options)
    # # Get initial parameters (for comparison purposes)
    # with tf.Session():
    #     tf.initialize_all_variables().run()
    #     _save_params(params, model_options['fp_params'])
    #     input('saved')
    softmax_layer = layers[-1]

    xent = -tf.reduce_sum(y * tf.log(softmax_layer))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(xent)

    correct = tf.equal(tf.argmax(y, 1), tf.argmax(softmax_layer, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    sess = tf.Session()

    with sess:
        try:
            tf.initialize_all_variables().run()

            for i in range(model_options['max_train_steps']):
                batch = data.train.next_batch(model_options['batch_size'])

                if i % 100 == 0:
                    # batch_accuracy = accuracy.eval(feed_dict={
                    #     x: batch[0], y: batch[1]
                    # })
                    # logger.info("step %d, training accuracy %g", i, batch_accuracy)
                    batch_cost = xent.eval(feed_dict={
                        x: batch[0], y: batch[1]
                    })
                    logger.info("step %d, training cost %g", i, batch_cost)

                train_step.run(feed_dict={x: batch[0], y: batch[1]})

        except KeyboardInterrupt:
            logger.info('Stopping training')
        finally:
            _save_params(params, model_options['fp_params'])

        test_accuracy = accuracy.eval(feed_dict={x: data.test.images, y: data.test.labels})
        logger.info("test accuracy %g", test_accuracy)


### PARAMS

FP_PARAMS = 'params_norm3.pkl'
BATCH_SIZE = 50
IMAGE_DIM_SIZE = 28
NUM_CLASSES = 10
NUM_FC_UNITS = 1024
MAX_TRAIN_STEPS = 20000

conv_layers = [
    {'filter': [5, 5, 1, 32], 'strides': [1] * 4, 'padding': 'SAME', 'act_fn': tf.nn.relu},
    {'filter': [5, 5, 32, 64], 'strides': [1] * 4, 'padding': 'SAME', 'act_fn': tf.nn.relu},
]

pool_layers = [
    {'filter': [1, 2, 2, 1], 'strides': [1, 2, 2, 1], 'padding': 'SAME'},
    {'filter': [1, 2, 2, 1], 'strides': [1, 2, 2, 1], 'padding': 'SAME'},
]

assert len(conv_layers) == len(pool_layers)

model_options = {'image_dim_size': IMAGE_DIM_SIZE,
                 'num_features': IMAGE_DIM_SIZE ** 2,
                 'fp_params': FP_PARAMS,
                 'conv_layers': conv_layers,
                 'pool_layers': pool_layers,
                 'num_fc_units': NUM_FC_UNITS,
                 'num_classes': NUM_CLASSES,
                 'batch_size': BATCH_SIZE,
                 'max_train_steps': MAX_TRAIN_STEPS,
                 }


if __name__ == '__main__':

    FORMAT = '%(levelname)s: %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=FORMAT,
    )

    ### PROCESS

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    train(mnist, model_options)

