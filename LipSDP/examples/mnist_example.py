import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
from torchsummary import summary
from MNIST_Net import Network
from scipy.io import savemat
import numpy as np
import os

INPUT_SIZE = 784
OUTPUT_SIZE = 10
BATCH_SIZE = 100
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

def main():

    train_loader, test_loader = create_data_loaders()
    fname = os.path.join(os.getcwd(), 'saved_weights/mnist_weights.mat')

    # define neural network model and print summary
    net_dims = [INPUT_SIZE, 50, OUTPUT_SIZE]
    model = Network(net_dims, activation=nn.ReLU).net
    summary(model, (1, INPUT_SIZE))

    # train model
    accuracy = train_network(model, train_loader, test_loader)

    # save data to saved_weights/ directory
    weights = extract_weights(model)
    data = {'weights': np.array(weights, dtype=np.object)}
    savemat(fname, data)

def extract_weights(net):
    """Extract weights of trained neural network model

    params:
        * net: torch.nn instance - trained neural network model

    returns:
        * weights: list of arrays - weights of neural network
    """

    weights = []
    for param_tensor in net.state_dict():
        tensor = net.state_dict()[param_tensor].detach().numpy().astype(np.float64)

        if 'weight' in param_tensor:
            weights.append(tensor)

    return weights


def train_network(model, train_loader, test_loader):
    """Train a neural network with Adam optimizer

    params:
        * model: torch.nn instance            - neural network model
        * train_loader: DataLoader instance   - train dataset loader
        * test_loader: DataLoader instance    - test dataset loader

    returns:
        * accuracy: float - accuracy of trained neural network
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch_num in range(1, NUM_EPOCHS + 1):
        train_model(model, train_loader, optimizer, criterion, epoch_num)
        accurary = test_model(model, test_loader)

    return accurary

def create_data_loaders():
    """Create DataLoader instances for training and testing neural networks

    returns:
        * train_loader: DataLoader instance   - loader for training set
        * test_loader: DataLoader instance    - loader for test set
    """

    train_set = datasets.MNIST('/tmp', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    test_set = datasets.MNIST('/tmp', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, test_loader

def train_model(model, train_loader, optimizer, criterion, epoch_num, log_interval=200):
    """Train neural network model with Adam optimizer for a single epoch

    params:
        * model: nn.Sequential instance                 - NN model to be tested
        * train_loader: DataLoader instance             - Training data for NN
        * optimizer: torch.optim instance               - Optimizer for NN
        * criterion: torch.nn.CrossEntropyLoss instance - Loss function
        * epoch_num: int                                - Number of current epoch
        * log_interval: int                             - interval to print output

    modifies:
        weights of neural network model instance
    """

    model.train()   # Set model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(BATCH_SIZE, -1)

        optimizer.zero_grad()   # Zero gradient buffers
        output = model(data)    # Pass data through the network
        loss = criterion(output, target)    # Calculate loss
        loss.backward()     # Backpropagate
        optimizer.step()    # Update weights

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tCross-Entropy Loss: {:.6f}'.format(
                epoch_num, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def test_model(model, test_loader):
    """Test neural network model using argmax classification

    params:
        * model: nn.Sequential instance   - torch NN model to be tested
        * test_loader:                    - Test data for NN

    returns:
        * test_accuracy: float - testing classification accuracy
    """

    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.view(BATCH_SIZE, -1)
            output = model(data)

            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)               # Increment the total count
            correct += (predicted == labels).sum()     # Increment the correct count

    test_accuracy = 100 * correct.numpy() / float(total)
    print('Test Accuracy: %.3f %%\n' % test_accuracy)

    return test_accuracy

if __name__ == '__main__':
    main()
