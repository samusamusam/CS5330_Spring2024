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
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    // declare feature CSV file names
    const char FEATURE_7X7_CSV[] = "../features/feature7x7.csv";
    const char FEATURE_HIST_CSV[] = "../features/featureHist.csv";
    const char FEATURE_MULTI_HIST_CSV[] = "../features/featureMultiHist.csv";
    const char FEATURE_COLOR_TEXTURE_HIST_CSV[] = "../features/featureColorTextureHist.csv";
    const char FEATURE_RESNET_CSV[] = "../features/ResNet18_olym.csv";

    // intro to program
    cout << "Welcome to the image matching program." << endl;
    cout << "You may press 'q' at any time to exit the program." << endl;

    // store the return response variable for creating feature CSV files
    int createFeatureCSVFilesResponse = 1;
    char directory_path[256];

    while (createFeatureCSVFilesResponse != 0)
    {
        // have user select directory of images
        cout << "To get started, please first define the directory that your images are located in." << endl;
        cin.getline(directory_path, 256);
        // exit program if input is 'q'
        if (string(directory_path) == "q")
        {
            return -1;
        }
        cout << "Directory path set to: " << directory_path << endl;

        // create all feature csv files
        cout << "Creating all feature csv files for the directory of images..." << endl;
        createFeatureCSVFilesResponse = createFeatureCSVFiles(directory_path, FEATURE_7X7_CSV, FEATURE_HIST_CSV,
                                                              FEATURE_MULTI_HIST_CSV, FEATURE_COLOR_TEXTURE_HIST_CSV);

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

    // initialize matches
    vector<string> matches;

    for (;;)
    {
        // set variables for target image
        string targetImgName;
        string targetImgPath;
        Mat targetImg;
        targetImg.release(); // empty targetImg
        matches.clear();     // empty matches

        // ask user for target image in the set directory path
        do
        {
            cout << "Choose the target image in the directory: " + string(directory_path) + "." << endl;
            getline(cin, targetImgName);
            // exit program if input is 'q'
            if (targetImgName == "q")
            {
                return -1;
            }

            // set image full relative path
            targetImgPath = string(directory_path) + "/" + targetImgName;

            // attempt to read the targetImgName
            targetImg = imread(targetImgPath);

            // if failed to read
            if (!targetImg.data)
            {
                cerr << "Error: Unable to load the image. Please make sure you put in the correct path." << endl;
            }
        } while (!targetImg.data);

        // options for image matching
        int option = 0;
        string input;
        bool isValidInput = false;
        cout << "Choose from the following options for image matching." << endl;
        cout << "1. Baseline Matching" << endl;
        cout << "2. Histogram Matching" << endl;
        cout << "3. Multi-Histogram Matching" << endl;
        cout << "4. Texture/Color Matching" << endl;
        cout << "5. Deep Network Embeddings" << endl;
        cout << "Type 'q' to quit." << endl;

        // get user input on matching type
        do
        {
            cout << "Enter your choice: ";
            getline(cin, input);
            // exit program if input is 'q'
            if (input == "q")
            {
                return -1;
            }

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
            // exit program if input is 'q'
            if (input == "q")
            {
                return -1;
            }

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
        switch (option)
        {
        case 1: // Baseline Matching 7x7 middle filter
            features7x7Matching(targetImg, targetImgPath, numMatches, matches, FEATURE_7X7_CSV);
            break;
        case 2: // Histogram Matching
            featuresHistMatching(targetImg, targetImgPath, numMatches, matches, FEATURE_HIST_CSV);
            break;
        case 3: // Multi-Histogram Matching
            featuresMultiHistMatching(targetImg, targetImgPath, numMatches, matches, FEATURE_MULTI_HIST_CSV);
            break;
        case 4: // Texture-Color Matching
            featuresColorTextureMatching(targetImg, targetImgPath, numMatches, matches, FEATURE_COLOR_TEXTURE_HIST_CSV);
            break;
        case 5: // Deep Network Embeddings
            featuresDenMatching(targetImg, targetImgName, numMatches, matches, FEATURE_RESNET_CSV);
            break;
        }

        // if no matches found
        if (matches.size() < 0)
        {
            cerr << "No matches found. Please try again." << endl;
            continue;
        }

        // show target image
        cout << "Showing target image." << endl;
        namedWindow("Target - " + targetImgPath, WINDOW_NORMAL);
        imshow("Target - " + targetImgPath, targetImg);

        // show all matched images
        cout << "Showing all image matches." << endl;
        Mat matchedImg;
        for (int i = 0; i < matches.size(); i++)
        {
            matchedImg = imread(matches[i]);
            if (!matchedImg.empty())
            { // Check if the image was loaded successfully
                namedWindow("Matched Image " + to_string(i + 1) + " - " + matches[i], WINDOW_NORMAL);
                imshow("Matched Image " + to_string(i + 1) + " - " + matches[i], matchedImg);
            }
            else
            {
                cerr << "Error: Unable to load matched image at path: " << matches[i] << endl;
            }
        }
        waitKey(-1); // wait for keypress
        destroyAllWindows(); // destroy all windows // TODO: fix this
    }

    return (0);
}