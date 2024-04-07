# Samuel Lee
# CS 5330
# Spring 2024
# This program loads the network model and applies the model to custom inputs created by the user.

# import statements
import train_tutorial
import cv2
import os
import sys
import torch
from torchvision.transforms import ToTensor, Compose, Normalize
import matplotlib.pyplot as plt


# this function loads the network file and stores it in the model
def load_network_file(file_path):
    device = train_tutorial.set_device()
    model = train_tutorial.NeuralNetwork().to(device)
    model.device = device
    model_state_dict = torch.load(file_path)
    model.load_state_dict(model_state_dict)

    return model, device


# this function gets the tensor output values based on the network model
def get_predictions(num_predictions, images, labels, model, device):
    # store all tensor outputs
    list_outputs = []

    # get first 10 examples
    with torch.no_grad():
        for i in range(10):
            # add extra dimension for PyTorch processing
            image = images[i].unsqueeze(0).to(device)
            # get output tensor based on model and image input
            output = model(image)
            # store output in list of outputs
            list_outputs.append(output)
            # get label with highest value in output tensor
            predicted = torch.argmax(output, dim=1).item()
            # loop through each value in output to print the category label and weight
            for j, value in enumerate(output.squeeze()):
                print("Label {}: {:.2f}".format(j, value.item()))
            # print predicted and actual labels
            print("Predicted label: {}".format(predicted))
            print("Actual label: {}".format(labels[i]))

    return list_outputs


# this function plots the predictions in a figure
def plot_predictions(list_output, images, rows, cols):
    # figure to draw images on
    figure = plt.figure(figsize=(rows * cols, rows * cols), num="Sample Images")
    for i in range(1, rows * cols + 1):
        img = images[i - 1].squeeze()
        predicted = torch.argmax(list_output[i - 1]).item()
        figure.add_subplot(3, 3, i)

        plt.title(f"Prediction: {predicted}")
        plt.axis("off")
        plt.imshow(img, cmap="gray")
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

            # switch black and white colors
            _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

            # normalize intensities
            image = image / 255.0

            # get tensor image
            tensor_image = (
                torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
            )

            # run each image through model
            output = model(tensor_image)

            # get prediction
            predicted = torch.argmax(output, dim=1).item()

            # re-size image for output
            image = cv2.resize(image, (500, 500))

            # write prediction on image
            cv2.putText(
                image,
                "PREDICTED: " + str(predicted),
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
    # load network file
    model, device = load_network_file("model_weights.pth")
    # evaluation mode; not training mode
    model.eval()

    # get test dataset and loader
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    test_data = train_tutorial.load_data("data", False, True, transform, 1000)

    # get first batch
    data_iter = iter(test_data)
    images, labels = next(data_iter)

    # get first 10 examples predictions
    list_outputs = get_predictions(10, images, labels, model, device)

    # show first 9 examples
    plot_predictions(list_outputs, images, 3, 3)

    # show new images and predictions based on model
    prediction_model("input_images", model, device)

    return


if __name__ == "__main__":
    main(sys.argv)
