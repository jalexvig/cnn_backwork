from matplotlib import pyplot as plt
import pickle

def read_results(label):
    with open('results_{}.pkl'.format(label), 'rb') as f:
        res = pickle.load(f)
    return res

def view_results(label, start=-1):
    res = read_results(label)
    for vals, softmaxes in res[start:]:
        for idx, (val, softmax) in enumerate(zip(vals, softmaxes)):
            print(softmax[label])
            plt.imshow(val)
            plt.show()

if __name__ == '__main__':

    view_results(0)
