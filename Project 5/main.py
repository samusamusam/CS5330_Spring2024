# Samuel Lee
# CS 5330
# Spring 2024
# This program...

# import statements
import sys
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
import matplotlib.pyplot as plt

# class definitions
# class MyNetwork(nn.Module):
#     def __init__(self):
#         pass

#     # computes a forward pass for the network
#     # methods need a summary comment
#     def forward(self, x):
#         return x

# # useful functions with a comment for each function
# def train_network( arguments ):
#     return


# this function loads data from the dataset and returns it
def load_data(path, boolTrain, boolDownload, transform):
    data = datasets.MNIST(
        root=path, train=boolTrain, download=boolDownload, transform=transform
    )
    return data


# this function plots sample images from the data set
def plot_data(data, rows, cols):
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

    # Plot the first six digits from the test set
    plot_data(test_data, 2, 3)

    # main function code
    return


if __name__ == "__main__":
    main(sys.argv)
