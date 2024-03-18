/**
 * Samuel Lee
 * CS 5330
 * Spring 2024
 * This program calibrates a camera and stores the parameters in a file.
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

  // initialize variables to be used
  Size chessboardSize(9, 6);
  vector<Point2f> corner_set;
  bool cornersFound = false;
  Mat image;
  int imageCounter = 1;
  namedWindow("Video Feed", WINDOW_NORMAL);

  // data structure to hold data for calibration
  vector<Point3f> point_set;
  vector<vector<Point3f>> point_list;
  vector<vector<Point2f>> corner_list;

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

    // detect and display corners of chessboard
    getCorners(chessboardSize, corner_set, image, cornersFound);

    // see if there is a waiting keystroke
    char keyPressed = waitKey(1);

    // if user presses 's' and corners are found, save corner_set and point_set to corresponding lists
    if (keyPressed == 's' && cornersFound)
    {
      corner_list.push_back(corner_set);
      point_list.push_back(point_set);

      // store calibration image in image directory and save by nth image file name
      string imageFileName = "../calibration_directory/calibrationImage_" + to_string(imageCounter++) + ".png";
      imwrite(imageFileName, image);
      cout << "Saved calibration image in " << imageFileName << endl;

      cout << "Saved corners and world points to corresponding lists for calibration." << endl;
    }

    // if user presses 'c' and enough calibration images are saved, calibrate the camera
    if (keyPressed == 'c' && imageCounter >= 5)
    {
      // get camera matrix
      Mat cameraMat = Mat::eye(3, 3, CV_64F);
      cameraMat.at<double>(0, 2) = image.cols / 2;
      cameraMat.at<double>(1, 2) = image.rows / 2;

      vector<Mat> rotations, translations;             // variables to store resulting rotations and translations
      Mat distortionCoeffs = Mat::zeros(0, 0, CV_64F); // assuming no distortion

      // print pre-calibrated camera matrix
      cout << "Pre-calibrated camera matrix: " << endl
           << cameraMat << endl;

      // calibrate camera
      double reprojectionErr = calibrateCamera(point_list, corner_list, image.size(), cameraMat, distortionCoeffs, rotations, translations, CALIB_FIX_ASPECT_RATIO);

      // print post-calibration results
      cout << "Calibrated camera matrix: " << endl
           << cameraMat << endl;
      cout << "Reprojection error: " << reprojectionErr << endl;
      cout << "Distortion coefficients: " << endl
           << distortionCoeffs << endl;

      // save camera matrices and reprojection error to a file
      FileStorage fs("../calibrationData.yml", FileStorage::WRITE);
      fs << "camera_matrix" << cameraMat;
      fs << "distortion_coefficients" << distortionCoeffs;
      fs << "reprojection_error" << reprojectionErr;
      fs << "rotations" << rotations;
      fs << "translations" << translations;

      fs.release();
      cout << "Saved calibration data to file." << endl;
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