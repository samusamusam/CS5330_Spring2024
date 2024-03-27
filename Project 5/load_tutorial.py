# Samuel Lee
# CS 5330
# Spring 2024
# This program...

# import statements
import train_tutorial
import sys
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


# this function saves the network to a file
def load_network_file(model, file_path):
    model_state_dict = torch.load(file_path)
    model.load_state_dict(model_state_dict)


# main function
def main(argv):
    # load network file
    device = train_tutorial.set_device()
    model = train_tutorial.NeuralNetwork().to(device)
    model.device = device
    load_network_file(model, "model_weights.pth")
    model.eval()

    # get test dataset and loader
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    test_data = train_tutorial.load_data("data", False, True, transform, 1000)

    # process first 10 examples
    data_iter = iter(test_data)
    images, labels = next(data_iter)

    with torch.no_grad():
        for i in range(10):
            # add extra dimension for PyTorch processing
            image = images[i].unsqueeze(0).to(device)
            # get output tensor based on model and image input
            output = model(image)
            # get label with highest value in output tensor
            predicted = torch.argmax(output, dim=1).item()
            # loop through each value in output to print the category label and weight
            for j, value in enumerate(output):
                print("Label {}: {:.2f}".format(j, value.item()))
            # print predicted and actual labels
            print("Predicted label: {}".format(predicted))
            print("Actual label: {}".format(labels[i]))

    return


if __name__ == "__main__":
    main(sys.argv)
