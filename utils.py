import os
import pickle


def read_results(label, folderpath):
    filepath = os.path.join(folderpath, 'results_{}.pkl'.format(label))
    with open(filepath, 'rb') as f:
        res = pickle.load(f)
    return res


def get_data_label(data, label, num=None):

    # This assumes one-hot encoding

    labels = data.train.labels
    mask = labels[:, label] == 1

    data_label = data.train.images[mask][:num]

    return data_label
