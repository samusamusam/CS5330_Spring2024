# Samuel Lee
# CS 5330
# Spring 2024
# This program analyzes the first layer of the trained network model and applies
# each of the filters in the first convolution layer to the first input image.

# import statements
import cv2
import torch
import load_tutorial
import train_tutorial
import sys
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Compose, Normalize


# this function prints the model and first layer and visualizes the first layer via PyPlot
def printAndVisualizeFirstLayer(model):
    # shows network model and layers used
    print("Network Model")
    print("-----------------------------------------------------")
    print(model)
    # first layer
    print("First Layer")
    print("-----------------------------------------------------")
    weights = model.conv1.weight.detach().cpu().numpy()
    # set up figure and axes for subplots to visualize the weights
    figure, axes = plt.subplots(3, 4, figsize=(10, 8))
    axes = axes.flatten()  # used for linear iteration

    # store filters in array
    firstLayerFilters = []

    # iterate over each filter
    for i in range(len(weights)):
        axis = axes[i]
        # plot filter with color map 'viridis'
        axis.imshow(weights[i, 0], cmap="viridis", interpolation="none")

        # format subplot
        axis.set_title(f"Filter {i}")
        axis.axis("off")

        # print filter index, shape, and weights
        print("Filter index:", i)
        print("Shape:", weights[i, 0].shape)
        print(weights[i, 0])
        firstLayerFilters.append(weights[i, 0])

    # hide any unused subplots
    for i in range(len(weights), len(axes)):
        axes[i].axis("off")

    # show visualization
    plt.tight_layout()
    plt.show()

    return firstLayerFilters


# this function returns the first image of a dataloader
def getFirstImage(dataloader):
    images, labels = next(iter(dataloader))
    image = images[0].squeeze().cpu().detach().numpy()
    return image


# this function applies each of the first layer filters to an image
def applyFirstLayerIndividually(layer, image):
    # get each filter
    filters = layer.weight.cpu().detach().numpy()

    # store output images from applying filter
    results = []

    # apply each filter to the image and store the resulting image
    with torch.no_grad():
        for i in range(len(filters)):
            filter = filters[i, 0]
            result = cv2.filter2D(image, -1, filter)
            results.append(result)

    # plot filter and corresponding image
    figure, axes = plt.subplots(5, 4, figsize=(10, 8))
    axes = axes.flat

    # loop through each result and filter and plot them
    for i in range(len(results)):
        axis = axes[i]

        # plot filter for even axes
        filter_axis = axes[i * 2]
        filter_axis.imshow(filters[i, 0], cmap="gray")
        filter_axis.axis("off")

        # plot resulting image for odd axes
        result_axis = axes[i * 2 + 1]  # odd index for results
        result_axis.imshow(results[i], cmap="gray")
        result_axis.axis("off")

    # show figure
    plt.tight_layout()
    plt.show()


# main function
def main(argv):
    # load network file
    model, device = load_tutorial.load_network_file("model_weights.pth")

    # print model and layers
    firstLayerFilters = printAndVisualizeFirstLayer(model)

    # load dataset
    transform = Compose(
        [
            ToTensor(),
            Normalize(
                (0.1307,),
                (0.3081,),
            ),
        ]
    )
    training_data = train_tutorial.load_data("data", True, True, transform, 64)

    # get first image
    firstImage = getFirstImage(training_data)

    # apply first layer filter to image based on number of filters and plot the result
    applyFirstLayerIndividually(model.conv1, firstImage)

    return


if __name__ == "__main__":
    main(sys.argv)
