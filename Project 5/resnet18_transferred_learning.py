# Samuel Lee
# CS 5330
# Spring 2024
# This program exhibits transferred learning by utilizing the pretrained resnet18 network model

# import statements
import torch
import sys
import cv2
import os
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, CenterCrop
from torch import nn
from PIL import Image


# transform class to prep data for resnet18 input
class LogoTransform:

    def __init__(self):
        # Initialize the transformations
        self.transform = Compose(
            [
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, x):
        return self.transform(x)


# this function prints all the layers of the model
def print_layers(model):
    # for each layer in the model, print it out
    for name, layer in model.named_children():
        print(name)
        print(layer)


# this function returns a model with the final fc layer replaced to categorize custom objects
def replace_final_fc_layer(model, categories):
    # freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # replace last layer
    model.fc = nn.Linear(512, categories)

    return model


# this function loads data from the image dataset and returns it
def load_data(path):
    # dataLoader for the logos data set
    # 0 = facebook; 1 = instagram; 2 = snapchat
    logo_train = DataLoader(
        ImageFolder(
            path,
            transform=Compose([LogoTransform()]),
        ),
        batch_size=3,
        shuffle=True,
    )

    return logo_train


# this function calculates the correctness in percentage of a model
def evaluateModel(model, dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
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
        # compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # back propogation
        loss.backward()
        optimizer.step()

        # print loss for each batch
        loss, current = loss.item(), batch * batch_size + len(X)
        train_losses.append(loss)
        train_counter.append((batch * batch_size) + (epoch_idx * size))
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# this function plots the train loss
def plot_train_data(train_losses, train_counter):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color="blue")
    plt.xlabel("number of training examples seen")
    plt.ylabel("loss")
    plt.show()


# this function reads example images that are not in the test or trained dataset and outputs predictions using the model
def prediction_model(images_directory, model):
    # loop through each image in directory
    files = os.listdir(images_directory)
    for file in files:
        if file.endswith("jpg") or file.endswith("jpeg") or file.endswith("png"):
            print("Reading image file named: ", file)
            # read each image
            image = cv2.imread(os.path.join(images_directory, file))

            # convert BGR to RGB
            tensor_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # convert the numpy array to PIL for torchvision transforms function
            tensor_image = Image.fromarray(tensor_image)

            # Define the transformation
            transform = LogoTransform()

            # Apply the transformation
            tensor_image = transform(tensor_image).unsqueeze(0)

            # run each image through model
            output = model(tensor_image)

            print(output)

            # get prediction
            predicted = torch.argmax(output, dim=1).item()

            # re-size image for output
            image = cv2.resize(image, (500, 500))

            # logo name assign
            logo_name = ""
            if predicted == 0:
                logo_name = "FACEBOOK"
            elif predicted == 1:
                logo_name = "INSTAGRAM"
            elif predicted == 2:
                logo_name = "SNAPCHAT"

            # write prediction on image
            cv2.putText(
                image,
                "PREDICTED: " + logo_name,
                (25, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )

            # show image
            cv2.imshow("Prediction", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows


# this function converts all images to a certain size
def convert_image_size(images_directory, size):
    # loop through each image in directory
    files = os.listdir(images_directory)
    for file in files:
        if file.endswith("jpg") or file.endswith("jpeg") or file.endswith("png"):
            # full file path
            full_path = os.path.join(images_directory, file)

            # read image
            image = cv2.imread(full_path)

            # re-size iamge
            image = cv2.resize(image, (size, size))

            cv2.imwrite(full_path, image)


# main function
def main(argv):

    # training variables
    num_epochs = 10
    batch_size = 3
    learning_rate = 0.001

    # load pretrained model
    model_resnet18 = models.resnet18(pretrained=True)

    # print layers of the pretrained model
    print_layers(model_resnet18)

    # replace final fc layer
    replace_final_fc_layer(model_resnet18, 3)

    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_resnet18.fc.parameters(), lr=learning_rate)

    # get images to train on
    dataloader = load_data("logos_training_images")

    # lists of training losses and counters
    train_losses = []
    train_counter = []

    # train model
    for t in range(num_epochs):
        print(f"Epoch #{t+1}\n-------------------------------")
        train_loop(
            dataloader,
            model_resnet18,
            loss_fn,
            optimizer,
            batch_size,
            train_losses,
            train_counter,
            t,
        )
        accuracy = evaluateModel(model_resnet18, dataloader)
        print(f"Epoch #{t+1}: Correctness: {accuracy*100:.2f}%")

    print("Finished training the model.")

    # plot the train loss data
    plot_train_data(train_losses, train_counter)

    # convert all images to 128x128
    convert_image_size("logos_test_images", 256)

    # use model to label new images
    prediction_model("logos_test_images", model_resnet18)

    # main function code
    return


if __name__ == "__main__":
    main(sys.argv)
