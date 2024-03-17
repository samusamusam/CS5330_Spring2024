/**
 * Samuel Lee
 * CS 5330
 * Spring 2024
 * Project to perform calibration and utilize augmented reality.
 */

#include "opencv2/opencv.hpp"
#include "detectCorners/detectCorners.h"
#include "calibration/calibration.h"

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

  // data structure to hold data for calibration
  vector<Vec3f> point_set;
  vector<vector<Vec3f>> point_list;
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
    
    // if user presses 's' and corners are found save corner_set and point_set to corresponding lists
    if (keyPressed == 's' && cornersFound) {
      corner_list.push_back(corner_set);
      point_list.push_back(point_set); 

      cout << "Saved corners and world points to corresponding lists for calibration." << endl;
    }

    // if user presses 'q' exit program
    if (keyPressed == 'q')
    {
      break;
    }
  }
  delete capdev;

  return (0);
}