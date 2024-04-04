# Samuel Lee
# CS 5330
# Spring 2024
# This program...

# import statements
import train_tutorial
import sys
import cv2
import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Normalize
import torch.nn.functional as F
import matplotlib.pyplot as plt


# greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


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
        self.fc2 = nn.Linear(50, 3)

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


# this function loads data from the image dataset and returns it
def load_data(path):
    # dataLoader for the Greek data set
    # 0 = alpha; 1 = beta; 2 = gamma
    greek_train = DataLoader(
        ImageFolder(
            path,
            transform=Compose(
                [
                    ToTensor(),
                    GreekTransform(),
                    Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=1,
        shuffle=True,
    )

    return greek_train


# this function retrieves pretrained weights from another similar network
def load_pretrained_weights():
    # get and set device
    device = train_tutorial.set_device()
    model = NeuralNetwork().to(device)
    model.device = device

    # load pretrained weights
    pretrained_weights = torch.load("model_weights.pth", map_location=device)

    # set last layer values to none/null
    pretrained_weights.pop("fc2.weight", None)
    pretrained_weights.pop("fc2.bias", None)

    # load pretrained weights minus the last layer
    model.load_state_dict(pretrained_weights, strict=False)

    # before replacing linear layer, freeze weights

    # freeze all network weights
    for name, param in model.named_parameters():
        if name in ["fc2.weight", "fc2.bias"]:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model, device


# this function calculates the correctness in percentage of a model
def evaluateModel(model, dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            # use same device as model
            X, y = X.to(model.device), y.to(model.device)
            # get prediction
            pred = model(X)
            # checks correct predictions of all values in tensor pred and adds count to variable correct
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    return correct / len(dataloader.dataset)


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

        # print loss for each batch
        loss, current = loss.item(), batch * batch_size + len(X)
        train_losses.append(loss / len(X))
        train_counter.append((batch * batch_size) + (epoch_idx * size))
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# this function plots the train loss
def plot_train_data(train_losses, train_counter):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color="blue")
    plt.xlabel("number of training examples seen")
    plt.ylabel("loss")  # FIND OUT WHAT LOSS THIS IS
    plt.show()


# this function reads example images that are not in the test or trained dataset and outputs predictions using the model
def prediction_model(images_directory, model, device):
    # loop through each image in directory
    files = os.listdir(images_directory)
    for file in files:
        if file.endswith("jpg") or file.endswith("jpeg") or file.endswith("png"):
            print("Reading image file named: ", file)
            # read each image
            image = cv2.imread(os.path.join(images_directory, file))

            # convert to gray image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # re-size to 28x28
            image = cv2.resize(image, (28, 28))

            # normalize intensities
            image = image / 255.0

            # get tensor image
            tensor_image = (
                torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
            )

            # run each image through model
            output = model(tensor_image)

            print(output)

            # get prediction
            predicted = torch.argmax(output, dim=1).item()

            # re-size image for output
            image = cv2.resize(image, (500, 500))

            # greek letter assign
            greek_letter = ""
            if predicted == 0:
                greek_letter = "alpha"
            elif predicted == 1:
                greek_letter = "beta"
            elif predicted == 2:
                greek_letter = "gamma"

            # write prediction on image
            cv2.putText(
                image,
                "PREDICTED: " + greek_letter,
                (25, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

            # show image
            cv2.imshow("Prediction", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows


# main function
def main(argv):
    # load pretrained network model
    model, device = load_pretrained_weights()

    # get dataloader
    dataloader = load_data("greek_letters")

    # training variables
    num_epochs = 30
    batch_size = 1
    learning_rate = 0.001

    # loss function and optimizer
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # lists of training losses and counters
    train_losses = []
    train_counter = []

    # train model
    for t in range(num_epochs):
        print(f"Epoch #{t+1}\n-------------------------------")
        train_loop(
            dataloader,
            model,
            loss_fn,
            optimizer,
            batch_size,
            train_losses,
            train_counter,
            t,
        )
        accuracy = evaluateModel(model, dataloader)
        print(f"Epoch #{t+1}: Correctness: {accuracy*100:.2f}%")

    print("Finished training the model.")

    # print model
    print(model)

    # plot the train loss data
    plot_train_data(train_losses, train_counter)

    # use model to label new images
    prediction_model("greek_letters_input_Images", model, device)

    # main function code
    return


if __name__ == "__main__":
    main(sys.argv)
