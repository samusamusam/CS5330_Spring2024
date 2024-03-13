/**
 * Samuel Lee

 * CS 5330
 * Spring 2024
 * Project to perform calibration and utilize augmented reality.
 */

#include "opencv2/opencv.hpp"

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
  Size chessboardSize(9,6);
  vector<Point2f> cornerSet;
  Mat image, grayImage;

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

    // initialize gray image
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    bool cornersFound = findChessboardCorners(grayImage, chessboardSize, cornerSet, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FILTER_QUADS + CALIB_CB_FAST_CHECK);

    // if corners found successfully
    if (cornersFound) {
      // variables for cornerSubPix
      Size searchArea(11,11);
      Size zeroZone(-1,-1);
      TermCriteria terminationCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1);

      // refine position of corners
      cornerSubPix(grayImage, cornerSet, searchArea, zeroZone, terminationCriteria);

      // draw corners
      drawChessboardCorners(image, chessboardSize, cornerSet, cornersFound);
    }

    // see if there is a waiting keystroke
    char keyPressed = waitKey(1);
    if (keyPressed == 'q')
    {
      break;
    }

  }
  delete capdev;

  return (0);
}