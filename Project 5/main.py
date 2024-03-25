# Samuel Lee
# CS 5330
# Spring 2024
# This program...

# import statements
import sys
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
import torch.nn.functional as F
import matplotlib.pyplot as plt


# this class represents a custom neural network
# given a 28x28 input image
# 5x5 convolution layer reduces image to 24x24
#
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # convolution layer with 10 5x5 filters
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # convolution layer with 20 5x5 filters with 10 channels as input
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # dropout layer with 50%  dropout rate
        self.drop50 = nn.Dropout2d(p=0.5)
        # flattening operation
        self.flatten = nn.Flatten()
        # linear layers
        # fc1 has 320 as an input because of how convolution layers and maxpool layers change the feature maps
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # applies convolution layer first, then max pooling layer with a 2x2 window, then ReLU
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # applies second convolution layer, dropout 50%, max pooling, ReLU in that order
        x = F.relu(F.max_pool2d(self.drop50(self.conv2(x)), 2))
        # flattens layer
        x = self.flatten(x)
        # apply linear layer 1 50 nodes then ReLU
        x = F.relu(self.fc1(x))
        # apply linear layer 2
        x = self.fc2(x)
        # return log softmax of output
        return F.log_softmax(x, dim=1)


# this function loads data from the dataset and returns it
def load_data(path, boolTrain, boolDownload, transform):
    # loads dataset
    data = datasets.MNIST(
        root=path, train=boolTrain, download=boolDownload, transform=transform
    )
    return data


# this function sets device to the optimal one
def set_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using {device} device")

    return device


# this function plots sample images from the data set
def plot_data(data, rows, cols):
    # creates figure where subplots will be located
    figure = plt.figure(figsize=(cols * 3, rows * 2), num="Sample Images")
    for i in range(1, cols * rows + 1):
        sample_idx = i - 1  # get sequentials samples without randomizing
        img, label = data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


# main function
def main(argv):
    # handle any command line arguments in argv

    # load MNIST test dataset
    transform = Compose([ToTensor()])
    test_data = load_data("data", True, True, transform)

    # plot the first 6 digits from test data set
    plot_data(test_data, 2, 3)

    # set device
    set_device()

    # main function code
    return


if __name__ == "__main__":
    main(sys.argv)
