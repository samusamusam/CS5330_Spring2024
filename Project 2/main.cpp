/**
 * Sam Lee
 * Spring 2024
 * CS5330
 */

#include "opencv2/opencv.hpp"
#include "compute_feature/compute_feature.h"
#include <cstdio>
#include <cstring>

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
    string imageMatchingOption;
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
    cout << "Choose the target image by specifying the full path." << endl;
    getline(cin, targetImgPath);
    targetImg = imread(targetImgPath); // attempt to read the targetImgPath
    if(!targetImg.data()) {
        
    }


    // get user input
    do
    {
        cout << "Enter your choice (1-10): ";
        getline(cin, imageMatchingOption);

        // try to convert string to integer
        stringstream(imageMatchingOption) >> option;

        // if input can be converted to an integer and the option inputted is between 1 and 10
        if (stringstream(imageMatchingOption) >> option && option >= 1 && option <= 10)
        {
            isValidInput = true;
        }
        else
        {
            cout << "Invalid input. Please enter a number between 1-10." << endl;
        }
    } while (!isValidInput);

    // switch cases based on user input
    switch (key)
    {
    case 1: // Baseline Matching 7x7 middle filter
        
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

    return (0);
}