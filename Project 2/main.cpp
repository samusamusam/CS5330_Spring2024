/**
 * Sam Lee
 * Spring 2024
 * CS5330
 */

#include "opencv2/opencv.hpp"
#include "compute_feature/compute_feature.h"
#include "image_match/image_match.h"
#include <cstdio>
#include <cstring>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{

    cout << "Welcome to the image matching program." << endl;
    cout << "You may press 'q' at any time to exit the program." << endl; // TODO: implement this

    // have user select directory of images
    cout << "To get started, please first define the directory that your images are located in." << endl;
    string directory_path;
    getline(cin, directory_path);
    cout << "Directory path set to: " << directory_path << endl;

    // store the return response variable for creating feature CSV files
    int createFeatureCSVFilesResponse = 1;

    while (createFeatureCSVFilesResponse != 0)
    {
        // create all feature csv files
        cout << "Creating all feature csv files for the directory of images..." << endl;
        int createFeatureCSVFilesResponse = createFeatureCSVFiles(directory_path);
        if (createFeatureCSVFilesResponse == -1)
        {
            cout << "Directory could not be opened. Try entering a new directory." << endl;
        }
        else if (createFeatureCSVFilesResponse == -2)
        {
            cout << "No image files to process in the directory. Try entering a new directory." << endl;
        }
    }

    // successfully created all CSV feature files
    cout << "Finished creating CSV files for all features." << endl;

    // options for image matching
    string input;
    int option = 0;
    bool isValidInput = false;
    cout << "Choose from the following options for image matching." << endl;
    cout << "1. Baseline Matching" << endl;
    cout << "2. " << endl;
    cout << "3. " << endl;
    cout << "4. " << endl;
    cout << "5. " << endl;

    // ask user for target image
    string targetImgPath;
    Mat targetImg;
    do
    {
        cout << "Choose the target image by specifying the full path." << endl;
        getline(cin, targetImgPath);
        targetImg = imread(targetImgPath); // attempt to read the targetImgPath
        if (!targetImg.data())
        {
            cerr << "Error: Unable to load the image. Please make sure you put in the correct path." << endl;
        }
    } while (!targetImg.data());

    // get user input on matching type
    do
    {
        cout << "Enter your choice (1-10): ";
        getline(cin, input);

        // try to convert string to integer
        stringstream(input) >> option;

        // if input can be converted to an integer and the option inputted is between 1 and 10
        if (stringstream(input) >> option && option >= 1 && option <= 10)
        {
            isValidInput = true;
        }
        else
        {
            cout << "Invalid input. Please enter a number between 1-10." << endl;
        }
    } while (!isValidInput);

    // reset input variables
    input = "";
    int numMatches = 0;
    isValidInput = false;

    // get number of matches user input
    do
    {
        cout << "How many matches do you want (between 3 and 5): ";
        getline(cin, input);

        // try to convert string to integer
        stringstream(input) >> numMatches;

        // if input can be converted to an integer and the option inputted is between 3 and 5
        if (stringstream(input) >> numMatches && numMatches >= 3 && numMatches <= 5)
        {
            isValidInput = true;
        }
        else
        {
            cout << "Invalid number of matches. Please enter a number between 3-5." << endl;
        }
    } while (!isValidInput);

    // switch cases based on user input
    vector<string> matches;
    switch (option)
    {
    case 1: // Baseline Matching 7x7 middle filter
        feature7x7Matching(targetImg, targetImgPath, numMatches, matches);
        break;
    case 2:

        break;
    case 3:

        break;
    case 4:

        break;
    case 5:

        break;
    }

    // if no matches found
    if(matches.size() < 0) {
        cerr << "No matches found. Please try again." << endl;
    }

    // show target image
    cout << "Showing target image." << endl;
    namedWindow("Target - " + targetImgPath, WINDOW_NORMAL);
    imshow("Target - " + targetImgPath, targetImg);

    // show all matched images
    cout << "Showing all image matches." << endl;
    Mat matchedImg;
    for(int i = 0; i < numMatches; i++) {
        matchedImg = imread(matches[i]);
        if (!matchedImg.empty()) { // Check if the image was loaded successfully
            namedWindow("Matched Image " + to_string(i+1), WINDOW_NORMAL);
            imshow("Matched Image " + to_string(i+1), matchedImg);
        }
        else {
            cerr << "Error: Unable to load matched image at path: " << matches[i] << endl;
        }
    }

    return (0);
}