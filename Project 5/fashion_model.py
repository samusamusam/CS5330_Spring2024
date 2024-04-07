# Samuel Lee
# CS 5330
# Spring 2024
# This program computes the accuracy and time taken to train a model based on model variables
# such as convolution filter size, convolution filter count, dropout rate, and more

# import statements
import train_tutorial
import sys
import time
import torch
import pandas as pd
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize
import torch.nn.functional as F


# this class represents a custom neural network
class CustomNetwork(nn.Module):
    def __init__(self, conv_filter_size, conv_filter_count, dropout_rate):
        super().__init__()
        # convolution layer
        self.conv = nn.Conv2d(1, conv_filter_count, kernel_size=conv_filter_size)
        # dropout layer
        self.dropout = nn.Dropout2d(p=dropout_rate)
        # flattening operation
        self.flatten = nn.Flatten()
        # linear layer
        self.fc = nn.Linear(
            conv_filter_count * (((28 - conv_filter_size + 1) // 2) ** 2), 10
        )

    def forward(self, x):
        # applies convolution layer first, then max pooling layer with a 2x2 window, then ReLU
        x = F.relu(F.max_pool2d(self.conv(x), 2))
        # applies dropout
        x = self.dropout(x)
        # applies flattening
        x = self.flatten(x)
        # applies linear layer
        x = self.fc(x)
        # return network
        return F.log_softmax(x, dim=1)


# this function loads data from the dataset and returns it
def load_data(path, boolTrain, boolDownload, transform, batch_size):
    # loads dataset
    data = datasets.FashionMNIST(
        root=path, train=boolTrain, download=boolDownload, transform=transform
    )

    # returns DataLoader with dataset
    return DataLoader(data, batch_size=batch_size, shuffle=boolTrain)


# this function trains and tests the model
def train_test_model(model, train_loader, test_loader, num_epochs):
    # define optimizer and loss function
    momentum = 0.5
    learning_rate = 0.001
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # start training
    for t in range(num_epochs):
        print(f"Epoch #{t+1}")
        train_loop(train_loader, model, loss_fn, optimizer)

    # get accuracy
    accuracy = test_loop(test_loader, model)

    return accuracy


# this function trains the model on training data
def train_loop(data_loader, model, loss_fn, optimizer):
    # get total dataset size
    size = len(data_loader.dataset)
    # sets model for training mode
    model.train()
    # loop through each batch in data_loader
    # X = tensor containing a batch of input images
    # y = tensor of labels of images
    for batch, (X, y) in enumerate(data_loader):
        # empty gradient since it accumulates
        optimizer.zero_grad()
        # use same device as model
        X, y = X.to(model.device), y.to(model.device)
        # compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # back propogation
        loss.backward()
        optimizer.step()


# this function checks the accuracy of the training model relative to test data
def test_loop(data_loader, model):
    # get total dataset size
    size = len(data_loader.dataset)
    # sets model for testing mode
    model.eval()
    # initialize test_loss and correct
    correct = 0

    # by using torch.no_grad(), no gradients are computed during test mode
    # this ensures that no unnecessary gradient computations are made for memory and speed efficiency
    with torch.no_grad():
        for X, y in data_loader:
            # use same device as model
            X, y = X.to(model.device), y.to(model.device)
            # get prediction
            pred = model(X)
            # checks correct predictions of all values in tensor pred and adds count to variable correct
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # get percent correct
    correct /= size
    correct *= 100

    return correct


# main function
def main(argv):
    # training/testing variables
    batch_size_test = 1000
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    # load MNIST test dataset into dataloaders
    test_data = load_data("fashion_data", False, True, transform, batch_size_test)

    # set device
    device = train_tutorial.set_device()

    # set dimensions of model
    conv_filter_size = [3, 5, 7]
    conv_filter_count = [5, 10, 15]
    dropout_rate = [0.3, 0.5, 0.7]
    epochs = [1, 3]
    batches = [32, 64]

    # store results
    results = []

    # loop through each of the dimensions by holding one constant
    for size_filter in conv_filter_size:
        for num_filter in conv_filter_count:
            for rate in dropout_rate:
                for epoch in epochs:
                    for batch in batches:
                        # settings used
                        print(
                            f"Settings: Filter Size: {size_filter}, Number of Filters: {num_filter}, Dropout Rate: {rate}, Epochs: {epoch}, Batch Size: {batch}"
                        )
                        print("-------------------------------")

                        # get training dataset
                        training_data = load_data(
                            "fashion_data", True, True, transform, batch
                        )

                        # initialize CNN
                        model = CustomNetwork(size_filter, num_filter, rate).to(device)
                        model.device = device

                        # start time
                        start_time = time.time()

                        # calculate accuracy
                        accuracy = train_test_model(
                            model, training_data, test_data, epoch
                        )

                        # end time
                        end_time = time.time()

                        # get difference in time
                        duration = end_time - start_time

                        # add result
                        settings = f"Filter Size: {size_filter}, Number of Filters: {num_filter}, Dropout Rate: {rate}, Epochs: {epoch}, Batch Size: {batch}"
                        results.append(
                            {
                                "Settings": settings,
                                "Duration": duration,
                                "Accuracy": accuracy,
                            }
                        )
                        print("-------------------------------")

    # store results in dataframe
    results_df = pd.DataFrame(results)

    # print results
    print(results_df.to_string())

    return


if __name__ == "__main__":
    main(sys.argv)
