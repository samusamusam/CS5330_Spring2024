/*
  Bruce A. Maxwell and Samuel Lee
  Spring 2024
  CS 5330 Computer Vision

  Include file for faceDetect.cpp, face detection and drawing functions
  Include file for faceDetect.cpp, eye detection function
*/
#ifndef FACEDETECT_H
#define FACEDETECT_H

// put the path to the haar cascade file here
#define FACE_CASCADE_FILE "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 1/faceDetect/haarcascade_frontalface_alt2.xml"
#define EYE_CASCADE_FILE "/Users/sam/Documents/Northeastern Khoury/CS5330 Pattern Recognition & Computer Vision/Projects/Project 1/faceDetect/haarcascade_eye_tree_eyeglasses.xml"

// prototypes
int detectFaces( cv::Mat &grey, std::vector<cv::Rect> &faces );
int detectEyes( cv::Mat &grey, std::vector<cv::Rect> &eyes );
int drawBoxes( cv::Mat &frame, std::vector<cv::Rect> &faces, int minWidth = 50, float scale = 1.0  );

#endif
