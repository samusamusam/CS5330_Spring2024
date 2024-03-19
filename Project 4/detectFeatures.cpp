/**
 * Samuel Lee
 * CS 5330
 * Spring 2024
 * This program detects robust features.
 */

#include "opencv2/opencv.hpp"
#include <iostream>

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

  // initialize variables to be used
  Mat image;
  namedWindow("Video Feed", WINDOW_NORMAL);

  // get some properties of the image
  Size refS((int)capdev->get(CAP_PROP_FRAME_WIDTH),
            (int)capdev->get(CAP_PROP_FRAME_HEIGHT));
  printf("Expected size: %d %d\n", refS.width, refS.height);

  // indefinite for loop that breaks based on key press
  for (;;)
  {
    *capdev >> image; // get a new frame from the camera, treat as a stream

    // Mat used for harris corners
    Mat resizedImage, gray, dst, dst_norm, dst_norm_scaled;

    // if no frame, exit
    if (image.empty())
    {
      cout << "frame is empty." << endl;
      return (-1);
    }
    
    // resize image for faster computation
    resize(image, image, Size(), 0.5, 0.5);

    // convert to grayscale
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // initialize empty Mat for dst
    dst = Mat::zeros(image.size(), CV_32FC1);

    // get harris corners
    cornerHarris(gray, dst, 7, 3, 0.06, BORDER_DEFAULT);

    // normalize harris corners Mat
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);

    // draw circles around each corner found
    for (int i = 0; i < dst_norm.rows; i++)
    {
      // row pointer
      float* rowPtr = dst_norm.ptr<float>(i);

      for (int j = 0; j < dst_norm.cols; j++)
      {
        // threshold for corner detection
        if (rowPtr[j] > 130)
        { 
          circle(image, Point(j, i), 5, Scalar(255, 0, 0), 2, 8, 0);
        }
      }
    }

    // see if there is a waiting keystroke
    char keyPressed = waitKey(1);

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