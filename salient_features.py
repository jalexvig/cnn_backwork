import tensorflow as tf
from proc_mnist import build_model, model_options
import numpy as np


def get_grads(model_options, x_vals, label):

    x_vals = x_vals.reshape((-1, 28 ** 2))
    params, x, y, layers = build_model(model_options)

    softmax_layer = layers[-1]
    softmax_label = softmax_layer[:, label]

    grads_input = tf.gradients(softmax_label, x)[0]

    with tf.Session():
        tf.initialize_all_variables().run()
        res = grads_input.eval(
            feed_dict={
                x: x_vals
            }
        )

        # print(softmax_label.eval(feed_dict={
        #     x: x_vals
        # }))

    return res


def get_salient_features(model_options, x_vals, label,
                         grad_sign='none', normalize=False, as_binary=False, num_keep=None, mul_weights=False):

    grads = get_grads(model_options, x_vals, label)
    weights = grads

    if grad_sign.lower()[:3] == 'pos':
        weights[weights < 0] = 0
    elif grad_sign.lower()[:3] == 'neg':
        weights[weights > 0] = 0

    weights = np.abs(weights)

    if num_keep is not None:
        partition = weights.shape[-1] - num_keep

        thresholds = np.partition(weights, partition)[:, partition]
        thresholds = np.repeat(thresholds[:, None], weights.shape[-1], axis=-1)

        weights[weights < thresholds] = 0

    if normalize:
        min_weight = weights[weights > 0].min(axis=-1)
        max_weight = weights[weights > 0].max(axis=-1)

        weights = ((weights.T - min_weight) / (max_weight - min_weight)).T
    elif as_binary:

        weights = (weights > 0).astype(float)

    if mul_weights:
        features = weights * x_vals.reshape((x_vals.shape[0], -1))
    else:
        features = weights

    return features


if __name__ == '__main__':

    model_options['fp_params'] = 'params/params_norm4.pkl'

    print(get_grads(model_options, np.random.rand(3, 28 ** 2), 2))
