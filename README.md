# LipSDP

This repository contains code for the Lipschitz constant estimation semidefinite programming framework introduced in [LipSDP](https://arxiv.org/abs/1906.04893) by Mahyar Fazlyab, Alexander Robey, Hamed Hassani, Manfred Morari, and George J. Pappas.  This work will appear as a conference paper at NeurIPS 2019.

Compared to other methods in this literature, this semidefinite programming approach for computing the Lipschitz constant of a neural network is both more scalable and accurate.  By viewing activation functions as gradients of convex potential functions, we use incremental quadratic constraints to formulate __LipSDP__, a convex program that estimates this Lipschitz constant.  We offer three forms of our SDP:
  1. __LipSDP-Network__ imposes constraints on all possible pairs of activation functions and has O(n²) decision variables. It is the least scalable but the most accurate method.
  2. __LipSDP-Neuron__ ignores the cross coupling constraints among different neurons and has O(n) decision variables. It is more scalable and less accurate than LipSDP-Network. For this case, we have T = diag(λ<sub>11</sub>,... , λ<sub>nn</sub>).
  3. __LipSDP-Layer__ considers only one constraint per layer, resulting in O(l) decision variables.  It is the most scalable and least accurate method. For this variant, we have T = blkdiag(λ<sub>1</sub>I<sub>n1</sub> ,... , λ<sub>m</sub>I<sub>nl</sub> ).
Furthermore, we offer support for using LipSDP-Network with random couplings of neurons.  We call this framework as __LipSDP-Network-Rand__.  

```
@inproceedings{fazlyab2019efficient,
  title     = {Efficient and Accurate Estimation of Lipschitz Constants for Deep Neural Networks},
  author    = {Fazlyab, Mahyar and Robey, Alexander and Hassani, Hamed and Morari, Manfred and Pappas, George J},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2019}
}
```
