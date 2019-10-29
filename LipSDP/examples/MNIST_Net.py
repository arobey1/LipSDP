import torch.nn as nn

class Network(nn.Module):
    def __init__(self, net_dims, activation=nn.ReLU):
        """Constructor for multi-layer perceptron pytorch class

        params:
            * net_dims: list of ints  - dimensions of each layer in neural network
            * activation: func        - activation function to use in each layer
                                      - default is ReLU
        """
        super(Network, self).__init__()

        layers = []
        for i in range(len(net_dims) - 1):
            layers.append(nn.Linear(net_dims[i], net_dims[i + 1]))

            # use activation function if not at end of layer
            if i != len(net_dims) - 2:
                layers.append(activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Pass data through the neural network model

        params:
            * x: torch tensor - data to pass though neural network

        returns:
            * ouput of neural network
        """

        return self.net(x)
