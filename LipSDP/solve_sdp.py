import argparse
import numpy as np
import matlab.engine
from scipy.io import savemat
import os

def main(args):

    eng = matlab.engine.start_matlab()
    eng.addpath(r'matlab_engine')

    network = {
        'alpha': matlab.double([args.alpha]),
        'beta': matlab.double([args.beta]),
        'net_dims': matlab.double([2, 10, 2]),
        'weight_path': os.path.join(os.getcwd(), 'test_weights'),
        'num_neurons': matlab.double([args.num_neurons])
    }

    lip_params = {
        'formulation': args.form,
        'split': False,
        'parallel': False,
        'verbose': args.verbose,
    }

    L = eng.solve_LipSDP(network, lip_params, nargout=1)
    print(L)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # formulations
    parser.add_argument('--form',
        default='neuron',
        const='neuron',
        nargs='?',
        choices=('neuron', 'network', 'layer', 'network-rand'),
        help='LipSDP formulation to use')

    parser.add_argument('-v', '--verbose',
        action='store_true',
        help='prints CVX output from solve if supplied')

    parser.add_argument('--alpha',
        type=int,
        default=0,
        nargs=1,
        help='lower bound for slope restriction bound')

    parser.add_argument('--beta',
        type=int,
        default=1,
        nargs=1,
        help='lower bound for slope restriction bound')

    parser.add_argument('--num-neurons',
        type=int,
        default=100,
        nargs=1,
        help="Number of neurons to couple for LipSDP-Network-rand formulation")

    args = parser.parse_args()
    main(args)
