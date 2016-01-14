from my_mnist_backwork_input_pool import train, model_options
import input_data
import logging
import numpy as np
import tensorflow as tf
import sys


### LOGGER SETTINGS
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
FORMAT = '%(levelname)s: %(message)s'
handler = logging.StreamHandler(sys.stderr)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


def augment_data(data):
    images = data.train.images
    labels = data.train.labels

    uniform_prob = np.empty((800, 10))
    uniform_prob.fill(1/10)

    images = np.concatenate((np.zeros((800, 784)), images))
    labels = np.concatenate((uniform_prob, labels))

    images = np.concatenate((np.random.rand(800, 784), images))
    labels = np.concatenate((uniform_prob, labels))

    images *= 255
    images = images.reshape((-1, 28, 28, 1))

    shuffled_indices = np.random.permutation(labels.shape[0])
    images = images[shuffled_indices]
    labels = labels[shuffled_indices]

    new_train = input_data.DataSet(images, labels)

    data.train = new_train

if __name__ == '__main__':

    model_options['fp_params'] = model_options['fp_params'].replace('norm', 'aug')

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    augment_data(mnist)

    train(mnist, model_options)
