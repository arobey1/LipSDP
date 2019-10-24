import argparse
import numpy as np
import matlab.engine
from scipy.io import savemat
import os



def main(args):

    eng = matlab.engine.start_matlab()
    eng.addpath(r'matlab_engine')

    weight_dict = {}
    weight_dict['0'] = matlab.double(np.random.rand(10, 2).tolist())
    weight_dict['1'] = matlab.double(np.random.rand(10, 2).tolist())
    alpha = 0
    beta = 1

    network = {
        'alpha': matlab.double([alpha]),
        'beta': matlab.double([beta]),
        'net_dims': matlab.double([2, 10, 2]),
        'weight_path': os.path.join(os.getcwd(), 'test_weights')
    }

    lip_params = {
        'formulation': args.form,
        'split': False,
        'parallel': False,
        'verbose': args.verbose,
    }

    L = eng.solve_LipSDP(network, lip_params, nargout=1)
    print(L)


def load_weights():
    data['network'] = {
        'weights': np.array(weights, dtype=np.object),
        'biases': np.array(biases, dtype=np.object),
        'dims': net_dims,
        'accuracy': acc,
    }

    savemat(file_name + '.mat', data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # formulations
    parser.add_argument('--form',
        default='neuron',
        const='neuron',
        nargs='?',
        choices=('neuron', 'network', 'layer'),
        help='LipSDP formulation to use')

    parser.add_argument('-v', '--verbose',
        action='store_true',
        help='prints CVX output from solve if supplied')

    parser.add_argument('-a', '--activation',
        default='relu',
        const='relu',
        nargs='?',
        choices='(relu, sigmoid, tanh, leaky-relu)',
        help='Activation in neural network model')

    args = parser.parse_args()
    main(args)
