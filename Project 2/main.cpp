/**
 * Sam Lee
 * Spring 2024
 * CS5330
 */

#include "opencv2/opencv.hpp"
#include "compute_feature/compute_feature.h"
#include "image_match/image_match.h"
#include "csv_util/csv_util.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    // declare feature CSV file names
    char FEATURE_7X7_CSV[] = "../features/feature7x7.csv";
    char FEATURE_HIST_CSV[] = "../features/featureHist.csv";
    char FEATURE_MULTI_HIST_CSV[] = "../features/featureMultiHist.csv";
    char FEATURE_COLOR_TEXTURE_HIST_CSV[] = "../features/featureColorTextureHist.csv";
    char FEATURE_RESNET_CSV[] = "../features/ResNet18_olym.csv";
    char FEATURE_COLOR_TEXTURE_DNN_HIST_CSV[] = "../features/featureColorTextureDNNHist.csv";
    char FEATURE_FACE_CSV[] = "../features/featureFace.csv";

    // intro to program
    cout << "Welcome to the image matching program." << endl;
    cout << "You may press 'q' at any time to exit the program." << endl;

    // store the return response variable for creating feature CSV files
    int createFeatureCSVFilesResponse = 1;
    char directory_path[256];

    // read pre-trained feature set
    vector<char *> imgFileNames;
    vector<vector<float>> imgFeatureData;
    read_image_data_csv("../features/ResNet18_olym.csv", imgFileNames, imgFeatureData, 0); // retrieve file names and feature data from ResNet csv
    while (createFeatureCSVFilesResponse != 0)
    {
        // have user select directory of images
        cout << "To get started, please first define the directory that your images are located in." << endl;
        cout << "Type 'q' to quit." << endl;
        cout << "     --------------------------------     " << endl;
        cin.getline(directory_path, 256);
        cout << "     --------------------------------     " << endl;
        // exit program if input is 'q'
        if (string(directory_path) == "q")
        {
            return -1;
        }
        cout << "Directory path set to: " << directory_path << endl;
        cout << "     --------------------------------     " << endl;

        // ask user for feature creation
        string inputFeature;
        cout << "Would you like to create the features? If the features are pre-loaded, you can skip this." << endl;
        cout << "'y' for create features; anything else for pre-loaded features" << endl;
        cout << "     --------------------------------     " << endl;
        getline(cin, inputFeature);
        cout << "     --------------------------------     " << endl;
        if (inputFeature != "y")
        {
            createFeatureCSVFilesResponse = 0;
        }
        if (inputFeature == "y")
        {
            // create all feature csv files
            cout << "Creating all feature csv files for the directory of images..." << endl;
            createFeatureCSVFilesResponse = createFeatureCSVFiles(directory_path, FEATURE_7X7_CSV, FEATURE_HIST_CSV,
                                                                  FEATURE_MULTI_HIST_CSV, FEATURE_COLOR_TEXTURE_HIST_CSV,
                                                                  FEATURE_COLOR_TEXTURE_DNN_HIST_CSV, FEATURE_FACE_CSV, imgFileNames, imgFeatureData);
            cout << "     --------------------------------     " << endl;

            if (createFeatureCSVFilesResponse == -1)
            {
                cout << "Directory could not be opened. Try entering a new directory." << endl;
                cout << "     --------------------------------     " << endl;
            }
            else if (createFeatureCSVFilesResponse == -2)
            {
                cout << "No image files to process in the directory. Try entering a new directory." << endl;
                cout << "     --------------------------------     " << endl;
            }
        }
    }

    // successfully created all CSV feature files
    cout << "Finished creating CSV files for all features." << endl;
    cout << "     --------------------------------     " << endl;

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
            matches.clear();     // empty matches
            cout << "Choose the target image in the directory: " + string(directory_path) + "." << endl;
            cout << "Type 'q' to quit." << endl;
            cout << "     --------------------------------     " << endl;
            getline(cin, targetImgName);
            cout << "     --------------------------------     " << endl;
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
                cout << "     --------------------------------     " << endl;
            }
        } while (!targetImg.data);
        for (;;)
        {
            // options for image matching
            matches.clear();     // empty matches
            int option = 0;
            string input;
            bool isValidInput = false;
            cout << "Choose from the following options for image matching." << endl;
            cout << "1. Baseline Matching" << endl;
            cout << "2. Histogram Matching" << endl;
            cout << "3. Multi-Histogram Matching" << endl;
            cout << "4. Texture/Color Matching" << endl;
            cout << "5. Deep Network Embeddings" << endl;
            cout << "6. Color/Texture/DNN Matching" << endl;
            cout << "7. Face Matching" << endl;
            cout << "Type 'b' to enter in a new target image." << endl;
            cout << "Type 'q' to quit." << endl;
            cout << "     --------------------------------     " << endl;

            // get user input on matching type
            do
            {
                cout << "Enter your choice: " << endl;
                cout << "     --------------------------------     " << endl;
                getline(cin, input);
                cout << "     --------------------------------     " << endl;
                // exit program if input is 'q'
                if (input == "q")
                {
                    return -1;
                }
                // exit do while to enter a new image if input is 'b'
                if (input == "b")
                {
                    break;
                }
                // try to convert string to integer
                stringstream(input) >> option;

                // if input can be converted to an integer and the option inputted is between 1 and 7
                if (stringstream(input) >> option && option >= 1 && option <= 7)
                {
                    isValidInput = true;
                }
                else
                {
                    cout << "Invalid input. Try again." << endl;
                    cout << "     --------------------------------     " << endl;
                }
            } while (!isValidInput);

            // exit for loop to enter a new image if input is 'b'
            if (input == "b")
            {
                break;
            }

            // reset input variables
            input = "";
            int numMatches = 0;
            isValidInput = false;

            // get number of matches user input
            do
            {
                cout << "How many matches do you want (between 1 and 5): " << endl;
                cout << "Type 'q' to quit." << endl;
                cout << "     --------------------------------     " << endl;
                getline(cin, input);
                cout << "     --------------------------------     " << endl;
                // exit program if input is 'q'
                if (input == "q")
                {
                    return -1;
                }

                // try to convert string to integer
                stringstream(input) >> numMatches;

                // if input can be converted to an integer and the option inputted is between 3 and 5
                if (stringstream(input) >> numMatches && numMatches >= 1 && numMatches <= 5)
                {
                    isValidInput = true;
                }
                else
                {
                    cout << "Invalid number of matches. Please enter a number between 3-5." << endl;
                    cout << "     --------------------------------     " << endl;
                }
            } while (!isValidInput);

            // switch cases based on user input
            vector<float> targetImgFeatureData;

            switch (option)
            {
            case 1: // Baseline Matching 7x7 middle filter
                feature7x7(targetImg, targetImgFeatureData);
                features_match_SSD(targetImg, targetImgName, numMatches, matches, FEATURE_7X7_CSV, targetImgFeatureData, false);
                break;
            case 2: // Histogram Matching
                featureHist(targetImg, targetImgFeatureData);
                features_match_intersection(targetImg, targetImgName, numMatches, matches, FEATURE_HIST_CSV, targetImgFeatureData, false);
                break;
            case 3: // Multi-Histogram Matching
                featureMultiHist(targetImg, targetImgFeatureData);
                features_match_intersection(targetImg, targetImgName, numMatches, matches, FEATURE_MULTI_HIST_CSV, targetImgFeatureData, false);
                break;
            case 4: // Texture-Color Matching
                featureColorTextureHist(targetImg, targetImgFeatureData);
                features_match_SSD(targetImg, targetImgName, numMatches, matches, FEATURE_COLOR_TEXTURE_HIST_CSV, targetImgFeatureData, false);
                break;
            case 5: // Deep Network Embeddings
                features_match_SSD(targetImg, targetImgName, numMatches, matches, FEATURE_RESNET_CSV, targetImgFeatureData, true);
                break;
            case 6: // Texture-Color-DNN Matching
                featureColorTextureDNNHist(targetImg, targetImgFeatureData, targetImgName, imgFileNames, imgFeatureData);
                features_match_SSD(targetImg, targetImgName, numMatches, matches, FEATURE_COLOR_TEXTURE_DNN_HIST_CSV, targetImgFeatureData, true);
                break;
            case 7: // Face Matching
                int response = featureFirstFace(targetImg, targetImgFeatureData);
                if(response == 0) {
                    features_match_SSD(targetImg, targetImgName, numMatches, matches, FEATURE_FACE_CSV, targetImgFeatureData, false);
                }
                break;
            }
            cout << "     --------------------------------     " << endl;

            // if no matches found
            if (matches.size() <= 0)
            {
                cerr << "No matches found. Please try again." << endl;
                cout << "     --------------------------------     " << endl;
                continue;
            }

            // show target image
            cout << "Showing target image." << endl;
            namedWindow("Target - " + targetImgName, WINDOW_NORMAL);
            imshow("Target - " + targetImgName, targetImg);

            // show all matched images
            cout << "Showing all image matches." << endl;
            cout << "     --------------------------------     " << endl;
            Mat matchedImg;
            for (int i = 0; i < matches.size(); i++)
            {
                matchedImg = imread(string(directory_path) + "/" + matches[i]);
                // Check if the image was loaded successfully
                if (!matchedImg.empty())
                {
                    namedWindow("Matched Image " + to_string(i + 1) + " - " + matches[i], WINDOW_NORMAL);
                    imshow("Matched Image " + to_string(i + 1) + " - " + matches[i], matchedImg);
                }
                else
                {
                    cerr << "Error: Unable to load matched image at path: " << matches[i] << endl;
                }
            }

            // keeps windows open until key press
            int keyPressed;

            // ask user to continue or quit the program
            cout << "Type 'c' if you would like to continue with the program." << endl;
            cout << "Type 'q' if you would like to quit the program." << endl;
            cout << "     --------------------------------     " << endl;
            do
            {
                keyPressed = waitKey(0);
                if (keyPressed == 'q')
                {
                    cout << "Thank you for using my Image Match program. See you again!" << endl;
                    return -1;
                }
                if (keyPressed == 'c')
                {
                    destroyAllWindows();
                    cout << "Type any key to confirm." << endl;
                    waitKey(0);
                }
            } while (keyPressed != 'q' && keyPressed != 'c');
            destroyAllWindows();
        }
    }
    return (0);
}