import numpy as np
from scipy.io import savemat
import os

def main():

    net_dims = [2, 10, 30, 20, 2]
    weights = create_rand_weights(net_dims)
    fname = os.path.join(os.getcwd(), 'saved_weights/random_weights.mat')

    data = {'weights': np.array(weights, dtype=np.object)}
    savemat(fname, data)


def create_rand_weights(net_dims):
    """Create random weights corresponding to given network layer dimensions

    params:
        * net_dims: list of ints - dimensions of each layer in network

    returns:
        * weights: list of arrays - weights of neural network
    """

    weights = []
    num_layers = len(net_dims) - 1

    for i in range(1, len(net_dims)):
        weights.append(1 / np.sqrt(num_layers) * np.random.rand(net_dims[i], net_dims[i-1]))

    return weights

if __name__ == '__main__':
    main()
