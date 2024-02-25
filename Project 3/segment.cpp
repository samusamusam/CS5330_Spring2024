/*
  Samuel Lee
  Anjith Prakash Chathan Kandy
  2/22/24
  This file contains functions to create a region map and color the region map.
*/

#include <unordered_set>
#include <cstdio>
#include <cstring>
#include <queue>
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

/**
 * This function creates a region map of the binary image.
 * src - binary image, cleaned up
 * minSize - minimum size of the region
 * stats - stats of the regions
 * num_labels - number of regions/labels
 * maxRegions - max regions to get from image
 * minDistances - keeps track of regions closest to center
 */
Mat labelConnectedComponents(Mat &src, int minSize, Mat &stats, int &num_labels, int maxRegions, priority_queue<pair<double, int>> &minDistances)
{
  // initialize variables
  Mat labels, centroids;

  // make PQ be ascending

  // get number of labels, stats, and labels
  num_labels = connectedComponentsWithStats(src, labels, stats, centroids);

  // check boundaries
  if (num_labels > stats.rows || num_labels > centroids.rows)
  {
    cerr << "error - number of labels should be <= number of rows " << endl;
    return labels;
  }

  // get center of image
  int centerX = src.cols / 2;
  int centerY = src.rows / 2;

  for (int i = 1; i < num_labels; i++)
  {
    bool touchesImgBoundary = false;                // whether region touches image boundary
    int labelArea = stats.at<int>(i, CC_STAT_AREA); // STAT_AREA is the total area of Connected Components

    int regionX = stats.at<int>(i, CC_STAT_LEFT);        // x-coordinate of the region
    int regionY = stats.at<int>(i, CC_STAT_TOP);         // y-coordinate of the region
    int regionWidth = stats.at<int>(i, CC_STAT_WIDTH);   // width of the region
    int regionHeight = stats.at<int>(i, CC_STAT_HEIGHT); // height of the region

    // checks if region touches boundary
    if (regionX <= 0 || regionY <= 0 || regionX + regionWidth >= src.cols || regionY + regionHeight >= src.rows)
    {
      touchesImgBoundary = true;
    }

    // if not touching the boundary and above minSize
    if (labelArea >= minSize && !touchesImgBoundary)
    // if (labelArea >= minSize)
    {
      // calculate centroid
      int centroidX = centroids.at<double>(i, 0);
      int centroidY = centroids.at<double>(i, 1);

      // calculate distance from center
      double distance = sqrt(((centroidX - centerX) * (centroidX - centerX)) + ((centroidY - centerY) * (centroidY - centerY)));

      minDistances.push({distance, i});

      // maintain PQ of min distance regions from center
      while (minDistances.size() > maxRegions)
      {
        minDistances.pop();
      }
    }
  }

  // update labels to keep only the region that meets the criteria
  for (int i = 1; i < num_labels; i++)
  {
    // check if region label is found in PQ
    bool found = false;
    priority_queue<pair<double, int>> tempQueue = minDistances; // copy of minDistances PQ
    while (!tempQueue.empty())
    {
      auto top = tempQueue.top();
      if (top.second == i)
      {
        found = true;
        break;
      }
      tempQueue.pop();
    }
    // if not found
    if (!found)
    {
      // set label to 0 (background)
      for (int j = 0; j < labels.rows; j++)
      {
        int *dptr = labels.ptr<int>(j);
        for (int k = 0; k < labels.cols; k++)
        {
          if (dptr[k] == i)
          {
            dptr[k] = 0;
          }
        }
      }
    }
  }
  return labels;
}

/**
 * This function creates a coloured region map of the binary image.
 * labels - labels from region map
 * colors - colors to represent each region
 * minDistances - keeps track of regions closest to center
 */

// first have map <Vec3b color, Point centroid> of size maxRegions; closest to center of image is color A, then color B, then color C ...
// compare centroids of label regions to the image centroid; whichever is closest is color A, then B, then C

Mat colorlabelConnectedComponents(Mat &labels, const vector<Vec3b> &colors, priority_queue<pair<double, int>> &minDistances)
{
  Mat colored(labels.size(), CV_8UC3); // image to store the colored version of the labels image

  // to make bg black
  Vec3b backgroundColor(0, 0, 0);

  // color the regions
  for (int i = 0; i < labels.rows; i++)
  {
    int *dptr = labels.ptr<int>(i);
    Vec3b *cptr = colored.ptr<Vec3b>(i);
    for (int j = 0; j < labels.cols; j++)
    {
      priority_queue<pair<double, int>> temp = minDistances; // create copy of minDistances
      int label = dptr[j];
      if (label == 0)
      {
        cptr[j] = backgroundColor; // set background to black
      }
      else
      {
        // find color associated with region
        bool found = false;
        int colorIndex = -1;
        while (!temp.empty())
        {
          auto top = temp.top();
          colorIndex++;
          if (top.second == label)
          {
            found = true;
            break;
          }
          temp.pop();
        }
        if (found)
        {
          cptr[j] = colors.at(colorIndex); // set found regions to a color
        }
        else
        {
          cptr[j] = backgroundColor; // set nonfound regions as black
        }
      }
    }
  }
  return colored;
}