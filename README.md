# Backworking convolutional nets

See [this post](http://jalexvig.github.io/blog/cnn-backwork/) for more details.

### proc_mnist.py

Train a convolutional neural network. Network topology may be specified in a dict passed to the train function.

Below is an example model configuration.

```python
FP_PARAMS = 'params_aug_misclass1.pkl'
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
```

And we can being training:

```python
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

train(mnist, model_options)
```

### cnn_backwork.py

Backwork out spurious yet strongly classified inputs.

Below is an example of additional model information that must be provided.

```python
num_layers = len(model_options['conv_layers']) + 3
cost_factors = np.logspace(-num_layers, -1, num=num_layers)
cost_factors[-1] *= 100

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
```

And to run the backworking model while saving pickled results to file:

```python
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

for label in range(10):

    model_options['label'] = label

    results = get_faulty_input_layer(mnist, model_options)
    with open('results_{}.pkl'.format(label), 'wb') as f:
        pickle.dump(results, f)
```

### preprocess_data.py

Augment the training data.

To augment with backworked data saved as results_i.pkl (for label i):

```python
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

augment_data(mnist)
```
