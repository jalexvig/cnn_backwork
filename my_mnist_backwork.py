import tensorflow as tf
import input_data
import logging
from collections import OrderedDict
import pickle
import os

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


def build_model(model_options, fp_params, const_params=False):

    num_features = model_options['num_features']
    num_fc_units = model_options['num_fc_units']
    conv_layer_channels = model_options['conv_layer_channels']

    if os.path.isfile(fp_params):
        params = load_params(fp_params, constant=const_params)
    else:
        params = init_params(num_features, conv_layer_channels, num_fc_units)

    layers = []

    x = tf.placeholder("float", shape=[None, IMAGE_DIM_SIZE ** 2])
    y = tf.placeholder("float", shape=[None, NUM_CLASSES])

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


def train(data, model_options, fp_params):

    params, x, y, layers = build_model(model_options, fp_params)
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
            save_params(params, fp_params)


def compute_layer_fixed_label(data, label, x, layer):

    labels = data.train.labels.argmax(1)
    mask = labels == label

    data_label = data.train.images[mask]

    fc_layer_label = layer.eval(feed_dict={
        x: data_label
    })

    return fc_layer_label


def get_inputs_from_conv_layer(model_options, params):

    num_features = model_options['num_features']
    params_list = list(params.values())

    image_flattened = tf.placeholder(tf.float32, shape=(1, num_features))

    dim_size = int(num_features ** .5)
    image = tf.reshape(image_flattened, (1, dim_size, dim_size))

    for W_conv, b_conv in params_list[-3::-1]:

        image -= b_conv
        image = tf.user_ops.deconvolve(image, W_conv)

    return image_flattened, image


def get_optimal_last_conv_layer(data, fp_params, model_options, label, num_opt_steps=8):

    num_features = model_options['num_features']

    params, x, y, layers = build_model(model_options, fp_params, const_params=True)
    params_list = list(params.values())
    last_conv_layer = layers[-3]

    with tf.Session():
        random_activations = tf.Variable(tf.random_normal((1, num_features), stddev=0.2))

        activations, image = get_inputs_from_conv_layer(model_options, params)

        tf.initialize_all_variables().run()

        W_fc, b_fc = params_list[-2]
        W_fc, b_fc = tf.constant(W_fc.eval()), tf.constant(b_fc.eval())
        fc_layer = tf.matmul(random_activations, W_fc) + b_fc

        W_softmax, b_softmax = params_list[-1]
        W_softmax, b_softmax = tf.constant(W_softmax.eval()), tf.constant(b_softmax.eval())
        softmax_layer = tf.nn.softmax(tf.matmul(fc_layer, W_softmax) + b_softmax)

        conv_layer_label = compute_layer_fixed_label(data, label, x, last_conv_layer).mean(axis=0)
        conv_layer_label = tf.constant(conv_layer_label)

        dist = tf.reduce_mean(tf.pow(conv_layer_label - random_activations, 2))

        cost = 1 / softmax_layer[0, label] + 1 / dist

        optimize_fc_layer_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
        tf.initialize_all_variables().run()

        for _ in range(num_opt_steps):
            optimize_fc_layer_step.run()

    return image, activations, random_activations


def backwork_inputs(data, fp_params, model_options, label=2, tol=0.001):

    image, activations, random_activations = get_optimal_last_conv_layer(data, fp_params, model_options, label)

    with tf.Session():
        tf.initialize_all_variables().run()
        input(random_activations.eval())
        image = image.eval(feed_dict={
            activations: random_activations.eval()
        })

    print(image, image.shape)


if __name__ == '__main__':

    FP_PARAMS = 'params_norm1.pkl'

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

    model_options = {'num_features': IMAGE_DIM_SIZE ** 2, 'conv_layer_channels': num_out_channels, 'num_fc_units': 1024}

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # train(mnist, model_options, FP_PARAMS)
    backwork_inputs(mnist, FP_PARAMS, model_options)
