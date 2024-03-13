/**
 * Samuel Lee
 * CS 5330
 * Spring 2024
 * Project to perform calibration and utilize augmented reality.
 */

#include "opencv2/opencv.hpp"
#include "detectCorners/detectCorners.h"

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
  Mat image;

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
    getCorners(chessboardSize, cornerSet, image);

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