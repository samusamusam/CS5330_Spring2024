/*
  Bruce A. Maxwell + Samuel Lee
  Spring 2024
  CS 5330 Computer Vision

  Functions for finding faces and drawing boxes around them - Bruce A. Maxwell
  Functions for finding eyes and their motions - Samuel Lee

  The paths to the Haar cascade files are defined in faceDetect.h
*/
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "faceDetect.h"
#include <algorithm>

// Comparator function for comparing two Rects by area
bool compareRectBySize(const cv::Rect &rect1, const cv::Rect &rect2)
{
  return (rect1.width * rect1.height) > (rect2.width * rect2.height);
}

/*
  Arguments:
  cv::Mat grey  - a greyscale source image in which to detect faces
  std::vector<cv::Rect> &faces - a standard vector of cv::Rect rectangles indicating where faces were found
     if the length of the vector is zero, no faces were found
 */
int detectFaces(cv::Mat &grey, std::vector<cv::Rect> &faces)
{
  // a static variable to hold a half-size image
  static cv::Mat half;

  // a static variable to hold the classifier
  static cv::CascadeClassifier face_cascade;

  // the path to the haar cascade file
  static cv::String face_cascade_file(FACE_CASCADE_FILE);

  if (face_cascade.empty())
  {
    if (!face_cascade.load(face_cascade_file))
    {
      printf("Unable to load face cascade file\n");
      printf("Terminating\n");
      exit(-1);
    }
  }

  // clear the vector of faces
  faces.clear();

  // cut the image size in half to reduce processing time
  cv::resize(grey, half, cv::Size(grey.cols / 2, grey.rows / 2));

  // equalize the image
  cv::equalizeHist(half, half);

  // apply the Haar cascade detector
  face_cascade.detectMultiScale(half, faces);

  // adjust the rectangle sizes back to the full size image
  for (int i = 0; i < faces.size(); i++)
  {
    faces[i].x *= 2;
    faces[i].y *= 2;
    faces[i].width *= 2;
    faces[i].height *= 2;
  }

  return (0);
}

/* Draws rectangles into frame given a vector of rectangles

   Arguments:
   cv::Mat &frame - image in which to draw the rectangles
   std::vector<cv::Rect> &faces - standard vector of cv::Rect rectangles
   int minSize - ignore rectangles with a width small than this argument
   float scale - scale the rectangle values by this factor (in case frame is different than the source image)
 */
int drawBoxes(cv::Mat &frame, std::vector<cv::Rect> &faces, int minWidth, float scale)
{
  // The color to draw, you can change it here (B, G, R)
  cv::Scalar wcolor(170, 120, 110);

  for (int i = 0; i < faces.size(); i++)
  {
    if (faces[i].width > minWidth)
    {
      cv::Rect face(faces[i]);
      face.x *= scale;
      face.y *= scale;
      face.width *= scale;
      face.height *= scale;
      cv::rectangle(frame, face, wcolor, 3);
    }
  }

  return (0);
}

/**
 * This function determines whether the eye region of interest is darker on the left side or right side
 * Arguments:
 * cv::Mat eyeROI - region of interest of the eye
 * cv::string text - text of whether eye is looking right or left
 */
void determineGazeDirection(const cv::Mat &eyeROI, std::string &text)
{
  // declare gradient
  cv::Mat gradientX;
  cv::Mat gradientY; 
  cv::Mat gradientMagnitude;

  cv::Sobel(eyeROI, gradientX, CV_64F, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT); // use the sobel method to get the horizontal gradient
  cv::Sobel(eyeROI, gradientY, CV_64F, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT); // use the sobel method to get the vertical gradient

  cv::magnitude(gradientX, gradientY, gradientMagnitude); // calculate magnitude of gradients

  cv::Rect leftHalf(0, 0, eyeROI.cols / 2, eyeROI.rows);                // gets the left half Rect
  cv::Rect rightHalf(eyeROI.cols / 2, 0, eyeROI.cols / 2, eyeROI.rows); // get the right half Rect

  gradientMagnitude.convertTo(gradientMagnitude, CV_8U); // convert gradient to 8U for thresholding

  cv::adaptiveThreshold(gradientMagnitude, gradientMagnitude, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, 1); // thresholding the gradient values

  // see if left side or right side is darker by summing up pixel values and normalizing them with the area
  double leftSum = cv::sum(gradientMagnitude(leftHalf))[0] / leftHalf.area();
  double rightSum = cv::sum(gradientMagnitude(rightHalf))[0] / rightHalf.area();

  // Compare sums for direction of eyes
  if (leftSum < rightSum)
  {
    text = "Looking left";
  }
  else
  {
    text = "Looking right";
  }
}

/**
 * This function detects eyes and draws circles around them
 * This function also tells the user whether the eyes are closed, looking left, looking right
 * This function detects which eye is closed as well
 * Arguments:
 * cv::Mat grey - an unsigned short grey image taken as a source
 * std::vector<cv::Rect> eyes - vector of eyes
 */
int detectEyes(cv::Mat &grey, std::vector<cv::Rect> &eyes)
{
  // declare faces and eyesHolder
  std::vector<cv::Rect> faces;
  std::vector<cv::Rect> eyesHolder;

  // get all faces detected
  detectFaces(grey, faces);

  // sort faces in descending order by size
  std::sort(faces.begin(), faces.end(), compareRectBySize);

  // a static variable to hold the classifier
  static cv::CascadeClassifier eye_cascade;

  // the path to the haar cascade file
  static cv::String eye_cascade_file(EYE_CASCADE_FILE);

  // text settings
  int fontFace = cv::FONT_HERSHEY_SIMPLEX;
  double fontScale = 1.0;
  int thickness = 3;
  cv::Scalar color(0, 0, 255);
  int baseline = 0;
  std::string text = "test";

  // if eye cascade is empty/failed to load
  if (eye_cascade.empty())
  {
    if (!eye_cascade.load(eye_cascade_file))
    {
      printf("Unable to load eye cascade file\n");
      printf("Terminating\n");
      exit(-1);
    }
  }

  // if no faces found
  if (faces.size() == 0)
  {
    text = "No face detected";
  }

  // if there are faces in the frame
  if (faces.size() > 0)
  {
    cv::Rect face = faces[0]; // get first face

    eyes.clear();                                   // clear previous eyes vector
    eye_cascade.detectMultiScale(grey, eyesHolder); // detect eyes on image

    // set eye size to detect min and max
    int minEyeWidth = static_cast<int>(0.10 * face.size().width);
    int maxEyeWidth = static_cast<int>(0.20 * face.size().width);

    // loop through each eyes
    for (int j = 0; j < eyesHolder.size(); j++)
    {
      // if eyes are detected according to the threshold of min and max eye size
      if (eyesHolder[j].width > minEyeWidth && eyesHolder[j].height > minEyeWidth &&
          eyesHolder[j].width < maxEyeWidth && eyesHolder[j].height < maxEyeWidth)
      {
        cv::Point eye_center(eyesHolder[j].x + eyesHolder[j].width / 2, eyesHolder[j].y + eyesHolder[j].height / 2); // find middle point of eyes
        if (face.contains(eye_center))
        {
          eyes.push_back(eyesHolder[j]);                                            // add to final eyes vector
          int radius = cvRound((eyesHolder[j].width + eyesHolder[j].height) * 0.5); // determine radius of circle around eyes
          circle(grey, eye_center, radius, cv::Scalar(255, 0, 0), 4);               // create circle around eyes
        }
      }
    }

    // if there is a face but no eyes, eyes are closed
    if (eyes.size() == 0)
    {
      text = "Eyes Closed";
    }

    if (eyes.size() == 1)
    {
      text = "Stop winking!";
    }

    if (eyes.size() == 2)
    {
      cv::Mat eyeROI1 = grey(eyes[0]); // get eye region of interest
      cv::Mat eyeROI2 = grey(eyes[1]); // get eye region of interest

      cv::Mat combinedEyeROI; // combined eye ROI

      // resize ROI
      cv::resize(eyeROI2, eyeROI2, eyeROI1.size());

      cv::addWeighted(eyeROI1, 0.5, eyeROI2, 0.5, 0, combinedEyeROI); // combine both ROIs weights
      // imshow("combined", combinedEyeROI); // show combined eye
      determineGazeDirection(combinedEyeROI, text); // determine which direction eyes are looking at
    }
  }

  // create text in frame
  cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
  cv::Point textOrg(grey.cols - textSize.width - 200, grey.rows - 50);
  cv::putText(grey, text, textOrg, fontFace, fontScale, color, thickness, cv::LINE_AA);
  return (0);
}
