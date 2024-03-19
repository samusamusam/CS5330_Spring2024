/**
 * Samuel Lee
 * CS 5330
 * Spring 2024
 * This program reads calibration data and an image/video and generates the AR object.
 */

#include "opencv2/opencv.hpp"
#include "detectCorners/detectCorners.h"
#include "calibration/calibration.h"
#include <iostream>
#include <vector>
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = std::__fs::filesystem;

/**
 * This function checks if the extension of the file path is a valid extension by checking with
 * the extensions in the vector<string> validExtensions
 * filePath - file path of the file to check
 * validExtensions - valid extensions
 * type - 0 is for image 1 is for video
 */
bool isValidExtension(const fs::path &filePath, const vector<string> &validExtensions, int &type)
{
  string fileExtension = filePath.extension().string(); // get extension
  transform(fileExtension.begin(), fileExtension.end(), fileExtension.begin(), [](unsigned char c)
            { return tolower(c); }); // make extension lowercase

  // loop through all valid extension to see if there is a match
  for (const string &extension : validExtensions)
  {
    if (fileExtension == extension)
    {
      if (filePath.extension() == ".mp4")
      {
        type = 1; // video
      }
      else
      {
        type = 0; // image
      }
      return true;
    }
  }
  return false;
}

int main(int argc, char *argv[])
{
  // read calibration data
  string calibrationDataFilePath = "../calibrationData.yml";
  FileStorage fs(calibrationDataFilePath, FileStorage::READ);
  if (!fs.isOpened())
  {
    cout << "Calibration data file does not exist. Please ensure you create this first." << endl;
    return -1;
  }

  Mat cameraMat, distortionCoeffs;
  fs["camera_matrix"] >> cameraMat;
  fs["distortion_coefficients"] >> distortionCoeffs;
  fs.release();

  // check if required data for program exist
  if (cameraMat.empty() || distortionCoeffs.empty())
  {
    cout << "The calibration data file does not have all the parameters needed to detect the chessboard." << endl;
    return -1;
  }

  // print post-calibration results
  cout << "Calibrated camera matrix: " << endl
       << cameraMat << endl;
  cout << "Distortion coefficients: " << endl
       << distortionCoeffs << endl;

  // initialize variables to be used
  Size chessboardSize(9, 6);
  vector<Point3f> point_set;
  vector<Point2f> corner_set;
  bool cornersFound = false;
  Mat image;
  VideoCapture video;

  // get chessboard world points and store it in point_set
  getChessboardWorldPoints(point_set, chessboardSize);

  // variables to check for user input and valid extension
  string input;
  vector<string> validExtensions = {".png", ".jpg", ".jpeg", ".mp4"};
  int type;

  // check if valid extension
  while (true)
  {
    cout << "Enter the path of the image or video you want to generate the image on (including the extension). Enter 'q' to quit." << endl;
    cin >> input;

    if (input == "q")
    {
      return -1;
    }

    if (isValidExtension(input, validExtensions, type))
    {
      break; // exit loop
    }
    else
    {
      cout << "Invalid file extension. Please enter a file with a valid extension (png, jpg, jpeg, mp4)." << endl;
    }
  }

  // check if valid image or video
  if (type == 0)
  { // for image
    image = imread(input);

    if (image.empty())
    {
      cerr << "Error: Image file could not be opened or read." << endl;
      return -1;
    }
    else
    {
      cout << "Image file read successfully." << endl;

      // detect and display corners of chessboard
      getCorners(chessboardSize, corner_set, image, cornersFound);

      // if corners are found
      if (cornersFound)
      {
        Mat rotations, translations;

        // get checkerboard's pose in terms of rotation and translation
        solvePnP(point_set, corner_set, cameraMat, distortionCoeffs, rotations, translations);

        cout << "Rotation:" << endl
             << rotations << endl;
        cout << "Translation:" << endl
             << translations << endl;

        // generate 3D object
        projectShape3D(rotations, translations, cameraMat, distortionCoeffs, image);

        // save image into a new file
        string newImageFilePath = "../generated_images/test.png";
        if (imwrite(newImageFilePath, image))
        {
          cout << "Image with AR saved successfully to " + newImageFilePath << endl;
          return 0;
        }
        else
        {
          cerr << "Image with AR did not save successfully." << endl;
          return -1;
        }
      }
      else
      {
        cout << "No chessboard target found. Exiting." << endl;
        return -1;
      }
    }
  }

  if (type == 1)
  { // for video
    if (!video.open(input))
    {
      cerr << "Error: Video file could not be opened or read." << endl;
      return -1;
    }
    else
    {
      cout << "Video file read successfully." << endl;

      // set up VideoWriter to output a video
      Size frameSize(static_cast<int>(video.get(CAP_PROP_FRAME_WIDTH)),
                     static_cast<int>(video.get(CAP_PROP_FRAME_HEIGHT)));
      double fps = video.get(CAP_PROP_FPS);
      VideoWriter outputVideo("../generated_images/test.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, frameSize, true);

      // check if output video opened
      if (!outputVideo.isOpened())
      {
        cerr << "Error opening the video to write on." << endl;
        return -1;
      }

      Mat frame; // variable to hold each frame in the video
      for (;;)
      {
        video >> frame;

        // checks if no more frames exist
        if (frame.empty())
        {
          break;
        }

        // detect and display corners of chessboard
        getCorners(chessboardSize, corner_set, frame, cornersFound);

        // if corners are found
        if (cornersFound)
        {
          Mat rotations, translations;

          // get checkerboard's pose in terms of rotation and translation
          solvePnP(point_set, corner_set, cameraMat, distortionCoeffs, rotations, translations);

          // generate 3D object
          projectShape3D(rotations, translations, cameraMat, distortionCoeffs, frame);
        }

        // add frame to video output
        outputVideo << frame;
      }
      video.release();
      outputVideo.release();
    }
    return (0);
  }
}