# Samuel Lee
# CS 5330
# Spring 2024
# This program...

# import statements
import sys
import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize
import torch.nn.functional as F
import matplotlib.pyplot as plt


# this class represents a custom neural network
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
def load_data(path, boolTrain, boolDownload, transform, batch_size):
    # loads dataset
    data = datasets.MNIST(
        root=path, train=boolTrain, download=boolDownload, transform=transform
    )

    # returns DataLoader with dataset
    return DataLoader(data, batch_size=batch_size, shuffle=boolTrain)


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
def plot_data(data_loader, rows, cols):
    # fetch first batch of iamges
    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    # creates figure where subplots will be located
    figure = plt.figure(figsize=(cols * 3, rows * 2), num="Sample Images")

    for i in range(1, cols * rows + 1):
        img = images[i - 1].squeeze()
        label = labels[i - 1].item()

        figure.add_subplot(rows, cols, i)
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.imshow(img, cmap="gray")
    plt.show()


# this function trains the model on training data
def train_loop(
    data_loader,
    model,
    loss_fn,
    optimizer,
    batch_size,
    train_losses,
    train_counter,
    epoch_idx,
):
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

        # print loss every 10 batches
        if batch % 10 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            train_losses.append(loss)
            train_counter.append((batch * batch_size) + (epoch_idx * size))
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# this function checks the accuracy/test error of the training model relative to test data
def test_loop(data_loader, model, loss_fn, test_losses):
    # get total dataset size
    size = len(data_loader.dataset)
    # sets model for testing mode
    model.eval()
    # initialize test_loss and correct
    test_loss, correct = 0, 0

    # by using torch.no_grad(), no gradients are computed during test mode
    # this ensures that no unnecessary gradient computations are made for memory and speed efficiency
    with torch.no_grad():
        for X, y in data_loader:
            # use same device as model
            X, y = X.to(model.device), y.to(model.device)
            # get prediction
            pred = model(X)
            # check loss between prediction of y based on model and actual y
            test_loss += loss_fn(pred, y).item()
            # checks correct predictions of all values in tensor pred and adds count to variable correct
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # get average loss and percent correct
    test_loss /= len(data_loader)
    correct /= size
    test_losses.append(test_loss)

    # print summary
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


# this function plots the train and test loss data
def plot_train_data(train_losses, test_losses, train_counter, test_counter):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color="blue")
    plt.scatter(test_counter, test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("number of training examples seen")
    plt.ylabel("loss")
    plt.show()


# this function saves the network to a file
def save_network_to_file(model, file_name):
    torch.save(model.state_dict(), file_name)


# main function
def main(argv):
    # training/testing variables
    num_epochs = 5
    momentum = 0.5
    batch_size = 64
    batch_size_test = 1000
    learning_rate = 0.001

    # load MNIST test dataset into dataloaders
    transform = Compose(
        [
            ToTensor(),
            Normalize(
                (0.1307,),
                (0.3081,),
            ),
        ]
    )
    training_data = load_data("data", True, True, transform, batch_size)
    test_data = load_data("data", False, True, transform, batch_size_test)

    # plot the first 6 digits from test data set
    plot_data(test_data, 2, 3)

    # set device
    device = set_device()

    # set network and move it to device
    network = NeuralNetwork().to(device)
    network.device = device

    # loss function and optimizer
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(
        network.parameters(), lr=learning_rate, momentum=momentum
    )

    # lists of training/test losses and counters
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(training_data.dataset) for i in range(num_epochs + 1)]

    # test model first as benchmark
    test_loop(test_data, network, loss_fn, test_losses)

    # train model
    for t in range(num_epochs):
        print(f"Epoch #{t+1}\n-------------------------------")
        train_loop(
            training_data,
            network,
            loss_fn,
            optimizer,
            batch_size,
            train_losses,
            train_counter,
            t,
        )
        test_loop(test_data, network, loss_fn, test_losses)

    print("Finished training the model.")

    # plot the train and test data
    plot_train_data(train_losses, test_losses, train_counter, test_counter)

    # save network model to file
    save_network_to_file(network, "model_weights.pth")

    # main function code
    return


if __name__ == "__main__":
    main(sys.argv)
