/**
 * Samuel Lee
 * CS 5330
 * Spring 2024
 * This program reads calibration data and performs augmented reality even if there are multiple targets.
 */

#include "opencv2/opencv.hpp"
#include "detectCorners/detectCorners.h"
#include "calibration/calibration.h"
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
  VideoCapture *capdev;

  // open the video device
  capdev = new VideoCapture(0);
  if (!capdev->isOpened())
  {
    printf("Unable to open video device\n");
    return (-1);
  }

  // get some properties of the image
  Size refS((int)capdev->get(CAP_PROP_FRAME_WIDTH),
            (int)capdev->get(CAP_PROP_FRAME_HEIGHT));
  printf("Expected size: %d %d\n", refS.width, refS.height);

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
  vector<vector<Point2f>> multipleCornerSet;
  Mat image;
  namedWindow("Video Feed", WINDOW_NORMAL);

  // get chessboard world points and store it in point_set
  getChessboardWorldPoints(point_set, chessboardSize);

  // indefinite for loop that breaks based on key press
  for (;;)
  {
    *capdev >> image; // get a new frame from the camera, treat as a stream

    // if no frame, exit
    if (image.empty())
    {
      cout << "frame is empty." << endl;
      return (-1);
    }

    // detect and display corners of all chessboards found
    getMultipleCorners(chessboardSize, multipleCornerSet, image);

    // see if there is a waiting keystroke
    char keyPressed = waitKey(1);

    // if corners are found
    if (multipleCornerSet.size() > 0) //CHECK THIS
    {
      // loop through each value of multipleCornerSet
      // in each loop
        // get rotations, translations via solvePnP, project 3D axes and project the shape
      Mat rotations, translations;

      // get checkerboard's pose in terms of rotation and translation
      solvePnP(point_set, corner_set, cameraMat, distortionCoeffs, rotations, translations);

      cout << "Rotation:" << endl
           << rotations << endl;
      cout << "Translation:" << endl
           << translations << endl;

      // project 3D axes
      project3DAxes(rotations,translations,cameraMat,distortionCoeffs,image);

      // project 3D object
      projectShape3D(rotations,translations,cameraMat,distortionCoeffs,image);
    }

    // if user presses 'q' exit program
    if (keyPressed == 'q')
    {
      break;
    }
    imshow("Video Feed", image);
  }
  delete capdev;

  return (0);
}