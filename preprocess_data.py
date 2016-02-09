import input_data
import numpy as np
from view_results import read_results


def _shuffle_images_labels(images, labels):

    shuffled_indices = np.random.permutation(labels.shape[0])
    images = images[shuffled_indices]
    labels = labels[shuffled_indices]

    return images, labels


def augment_random_data(data):
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

    images, labels = _shuffle_images_labels(images, labels)

    new_train = input_data.DataSet(images, labels)

    data.train = new_train


def augment_data(data, folderpath=''):

    images = data.train.images
    labels = data.train.labels

    images = images.reshape((-1, 28, 28, 1))
    imgs = [x for i in range(10) for x in read_results(i, folderpath)[-1][0]]
    imgs = np.array(imgs)[:, :, :, None]

    lbls = np.empty((imgs.shape[0], 10))
    lbls.fill(1/10)

    images = np.concatenate((imgs, images))
    labels = np.concatenate((lbls, labels))

    images, labels = _shuffle_images_labels(images, labels)

    new_train = input_data.DataSet(images, labels)

    data.train = new_train


if __name__ == '__main__':

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    augment_data(mnist)
