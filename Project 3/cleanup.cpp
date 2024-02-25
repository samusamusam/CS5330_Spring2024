/*
  Samuel Lee
  Anjith Prakash Chathan Kandy
  2/21/24
  This file  contains functions to apply cleanup via morphological filtering.
*/

#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

/**
 * This function applies an erosion to a greyscale image
 * src - image to perform erosion on
 * filterSize - dimension of the filter
 * cross - whether the filter is a cross; if false, it will default to square
 */
int erode(Mat &src, int filterSize, bool cross)
{
  // if no image is provided
  if (src.empty())
  {
    cerr << "Error: No input image provided." << endl;
    return -1;
  }

  // if filter size is invalid
  if (filterSize % 2 == 0 || filterSize < 3)
  {
    cerr << "Error: Invalid filter size." << endl;
    return -1;
  }

  // make a clone of src
  Mat temp = src.clone();

  // create filter
  vector<int> filterData;
  int center = filterSize / 2;

  if (cross)
  {
    filterData.assign(filterSize * filterSize, 0);
    // make all values in the cross 1
    for (int i = 0; i < filterSize; ++i)
    {
      if (i != center)
      {
        filterData[(i * filterSize) + center] = 1;
      }
      else
      {
        for (int j = 0; j < filterSize; ++j)
        {
          filterData[(i * filterSize) + j] = 1;
        }
      }
    }
  }
  else
  {
    filterData.assign(filterSize * filterSize, 1);
  }

  // loop through each pixel of the image
  for (int i = center; i < src.rows - center; i++)
  {
    // create vector of row pointers
    vector<uchar *> rowPtrs;
    for (int x = 0; x < filterSize; x++)
    {
      uchar *dptr = src.ptr<uchar>(i + x - center);
      rowPtrs.push_back(dptr);
    }
    for (int j = center; j < src.cols - center; j++)
    {
      int value = 255;

      // check if foreground pixel; then, loop through filter box and set value of current pixel
      if (rowPtrs.at(center)[j] == value)
      {
        for (int m = 0; m < filterSize * filterSize; m++)
        {
          if (filterData[m] == 1)
          {
            value = min(value, filterData[m] * rowPtrs.at(static_cast<int>(m / filterSize))[static_cast<int>(j - center + (m % filterSize))]);
            // break if foreground becomes background pixel
            if (value == 0)
            {
              break;
            }
          }
        }
      }
      else
      {
        value = static_cast<int>(rowPtrs.at(center)[j]); // original value if not foreground
      }
      temp.at<uchar>(i, j) = static_cast<uchar>(value); // assign value post-filter
    }
  }
  temp.copyTo(src);

  // release allocation of memory to temp
  temp.release();
  temp = Mat();

  return 0;
}

/**
 * This function applies a dilation to a greyscale image
 * src - image to perform erosion on
 * filterSize - dimension of the filter
 * cross - whether the filter is a cross; if false, it will default to square
 */
int dilate(Mat &src, int filterSize, bool cross)
{
  // if no image is provided
  if (src.empty())
  {
    cerr << "Error: No input image provided." << endl;
    return -1;
  }

  // if filter size is invalid
  if (filterSize % 2 == 0 || filterSize < 3)
  {
    cerr << "Error: Invalid filter size." << endl;
    return -1;
  }

  // make a clone of src
  Mat temp = src.clone();

  // create filter
  vector<int> filterData;
  int center = filterSize / 2;

  if (cross)
  {
    filterData.assign(filterSize * filterSize, 0);
    // make all values in the cross 1
    for (int i = 0; i < filterSize; ++i)
    {
      if (i != center)
      {
        filterData[(i * filterSize) + center] = 1;
      }
      else
      {
        for (int j = 0; j < filterSize; ++j)
        {
          filterData[(i * filterSize) + j] = 1;
        }
      }
    }
  }
  else
  {
    filterData.assign(filterSize * filterSize, 1);
  }

  // loop through each pixel of the image
  for (int i = center; i < src.rows - center; i++)
  {
    // create vector of row pointers
    vector<uchar *> rowPtrs;
    for (int x = 0; x < filterSize; x++)
    {
      uchar *dptr = src.ptr<uchar>(i + x - center);
      rowPtrs.push_back(dptr);
    }
    for (int j = center; j < src.cols - center; j++)
    {
      int value = 0;

      // check if foreground pixel; then, loop through filter box and set value of current pixel
      if (rowPtrs.at(center)[j] == value)
      {
        for (int m = 0; m < filterSize * filterSize; m++)
        {
          if (filterData[m] == 1)
          {
            value = max(value, filterData[m] * rowPtrs.at(static_cast<int>(m / filterSize))[static_cast<int>(j - center + (m % filterSize))]);
            // break if foreground becomes background pixel
            if (value == 255)
            {
              break;
            }
          }
        }
      }
      else
      {
        value = static_cast<int>(rowPtrs.at(center)[j]); // original value if not background
      }
      temp.at<uchar>(i, j) = static_cast<uchar>(value); // assign value post-filter
    }
  }
  temp.copyTo(src);

  // release allocation of memory to temp
  temp.release();
  temp = Mat();

  return 0;
}

/**
 * This function cleans up the thresholded image through morphological filters such as erosion and dilation.
 * thresholdedImage - image to manipulate
 */
void cleanup(Mat &src)
{
  // shrinking then dilating to reduce noise
  erode(src, 3, true);
  dilate(src, 3, false);
}