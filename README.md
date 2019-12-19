# LipSDP

This repository contains code for the Lipschitz constant estimation semidefinite programming framework introduced in [LipSDP](https://arxiv.org/abs/1906.04893) by Mahyar Fazlyab, Alexander Robey, Hamed Hassani, Manfred Morari, and George J. Pappas.  This work will appear as a conference paper at NeurIPS 2019.

Compared to other methods in this literature, this semidefinite programming approach for computing the Lipschitz constant of a neural network is both more scalable and accurate.  By viewing activation functions as gradients of convex potential functions, we use incremental quadratic constraints to formulate __LipSDP__, a convex program that estimates this Lipschitz constant.  We offer three forms of our SDP:
  1. __LipSDP-Network__ imposes constraints on all possible pairs of activation functions and has O(n²) decision variables, where `n` is the number of hidden neurons in the network. It is the least scalable but the most accurate method.
  2. __LipSDP-Neuron__ ignores the cross coupling constraints among different neurons and has O(n) decision variables. It is more scalable and less accurate than LipSDP-Network. For this case, we have T = diag(λ<sub>11</sub>,... , λ<sub>nn</sub>).
  3. __LipSDP-Layer__ considers only one constraint per layer, resulting in O(m) decision variables, where `m` is the number of hidden layers.  It is the most scalable and least accurate method. For this variant, we have T = blkdiag(λ<sub>1</sub>I<sub>n<sub>1</sub></sub> ,... , λ<sub>m</sub>I<sub>n<sub>m</sub></sub> ).

If you find this repository useful for your research, please consider citing our work:

```latex
@inproceedings{fazlyab2019efficient,
  title     = {Efficient and Accurate Estimation of Lipschitz Constants for Deep Neural Networks},
  author    = {Fazlyab, Mahyar and Robey, Alexander and Hassani, Hamed and Morari, Manfred and Pappas, George J},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2019}
}
```

## Installation and Requirements

After cloning this repository, it is easiest to set up a virtual environment to install the dependencies.  

```bash
python3 -m venv lipenv
```

Next, one can activate the environment and install the dependencies, which are listed in `requirements.txt`.

```bash
source activate lipenv/bin/activate
pip install -r requirements.txt
```

It is also necessary to install [CVX](http://cvxr.com/cvx/download/) for MATLAB.  Further, this repository uses the [MATLAB API for Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html), which also requires installation from the MATLAB command prompt.  Finally, to solve the semidefinite programs, we recommend the solver [MOSEK](https://www.mosek.com), which requires a license.  

## Usage

This package can be used to calculate Lipschitz constants of feed-forward neural networks.  To do so, a user must first save the weights of a neural network model in the ```.mat``` format.  Examples are given in the ```examples/``` directory.  

As a simple first example, consider the following code snippet which generates random weights in the following way:

```python
import numpy as np

weights = []
net_dims = [2, 10, 30, 20, 2]
num_layers = len(net_dims) - 1
norm_const = 1 / np.sqrt(num_layers)

for i in range(1, len(net_dims)):
  weights.append(norm_const * np.random.rand(net_dims[i], net_dims[i-1]))
```

This network has an input and output dimension of 2.  The hidden sizes are 10, 30, and then 20 neurons.  We next save the weights in the ```.mat``` format:

```python
from scipy.io import savemat

fname = os.path.join(os.getcwd(), 'saved_weights/random_weights.mat')
data = {'weights': np.array(weights, dtype=np.object)}
savemat(fname, data)
```

Then, to compute a Lipschitz constant for this network with the LipSDP-Neuron formulation, we can run the following command from the ```LipSDP/``` directory:

```bash
python solve_sdp.py --form neuron --weight-path examples/saved_weights/random_weights.mat
```

which should return something like

```
LipSDP-Neuron gives a Lipschitz constant of 36.482
```

A second example is provided in `examples/mnist_example.py`.  This script trains a one-hidden-layer neural network on the MNIST dataset using PyTorch.  The weights can be extracted and saved in a similar way as above.  Then to calculate a Lipschitz constant of the example MNIST weights in `examples/saved_weights/mnist_weights.mat` with the LipSDP-Layer, we can run

```bash
python solve_sdp.py --form layer --weight-path examples/saved_weights/mnist_weights.mat
```

## Formulations

There are several variants of the LipSDP semidefinite program introduced in this work.  If we let `n` be the number of neurons in the neural network and ```m``` be the number of hidden layers, then the scalability of our methods is as follows:

Formulation | LipSDP-Layer | LipSDP-Neuron | LipSDP-Network | LipSDP-Netowrk-Rand | LipSDP-Network-Dec-Vars
:---: | :---: | :---: | :---: | :---: | :--:
Scalability | O(m) | O(n) | O(n²) | O(k) | O(k)
`--form` argument | `layer` | `neuron` | `network` | `network-rand` | `network-dec-vars`


Here `k` is a user specified number (explained below).

Each formulation can be specified by supplying an argument with the ```--form``` flag.  The performance of these methods on ```examples/saved_weights/random_weights.mat``` is shown below:

```bash
export WEIGHT_PATH=examples/saved_weights/random_weights.mat

python solve_sdp.py --form layer --weight-path $WEIGHT_PATH
LipSDP-Layer gives a Lipschitz constant of 39.839

python solve_sdp.py --form neuron --weight-path $WEIGHT_PATH
LipSDP-Neuron gives a Lipschitz constant of 36.482

python solve_sdp.py --form network --weight-path $WEIGHT_PATH
LipSDP-Network gives a Lipschitz constant of 36.482
```

We also have several variants of LipSDP-Network that are implemented in this package.  In general, LipSDP-Network considers the couplings between each pair of decision variables; LipSDP-Network-Rand allows a user to consider a smaller subset of randomly chosen neuron couplings.  For example, to consider only 5 randomly chosen couplings from the (50 choose 2) = 1225 possible couplings, one can run

```bash
export WEIGHT_PATH=examples/saved_weights/mnist_weights.mat

python solve_sdp.py --form network-rand --num-neurons 5 --weight-path $WEIGHT_PATH
LipSDP-Network-rand gives a Lipschitz constant of 24.320
```

Another formulation is LipSDP-Network-Dec-Vars, which allows a user to specify the number of decision variables to be used in the semidefinite program.  An example with 5 decision variables is shown below:

```bash
python solve_sdp.py --form network-dec-vars --num-decision-vars 5 --weight-path $WEIGHT_PATH
LipSDP-Network-dec-vars gives a Lipschitz constant of 24.365
```

## Splitting

For larger networks, it is often efficacious to obtain a Lipschitz constant by solving to find Lipschitz constants for smaller subsets of layers and then multiplying the resulting constants.  To solve for a Lipschitz constant of the four-hidden-layer network saved in `examples/saved_weights/random_weights.mat` using LipSDP-Neuron using subnetworks with two layers each, we can run

```bash
export WEIGHT_PATH=examples/saved_weights/random_weights.mat

python solve_sdp.py --form neuron --split --split-size 2 --weight-path $WEIGHT_PATH
LipSDP-Neuron gives a Lipschitz constant of 37.668
```

Splitting can be used with any of the formulations given in the above table.  We used this splitting method to calculate Lipschitz constants for 100-hidden-layer neural networks with random weights.  The results are shown in Figure 2(b) of our paper.

To parallelize the solving of the SDPs corresponding to each subnetwork, add the `--parallel` flag.  Users can also optionally specify the number of workers using the `--num-workers` flag.  For example:

```bash
python solve_sdp.py --form neuron --split --split-size 2 --parallel --num-workers 6 --weight-path $WEIGHT_PATH
Starting parallel pool (parpool) using the 'local' profile ...
Connected to the parallel pool (number of workers: 6).
LipSDP-Neuron gives a Lipschitz constant of 37.668
```

## Activation Functions

As described in our paper, the semidefinite program used to compute Lipschitz constants depends on the choice of the activation function used in the neural network.  This choice can engender different values for the constants α and β.  For example, for ReLU, sigmoid, and tanh, α = 0 and β = 1.  To explicitly specify these constants, one can use the flags `--alpha` and `--beta` along the desired values.
