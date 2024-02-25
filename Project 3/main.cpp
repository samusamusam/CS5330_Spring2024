/**
 * Samuel Lee
 * Anjith Prakash Chathan Kandy
 * CS 5330
 * Spring 2024
 * Program to do 2D Image Recognition
 */

#include "opencv2/opencv.hpp"
#include <iostream>
#include "threshold.h"
#include "cleanup.h"
#include "segment.h"
#include "computeFeatures.h"
#include "csvReadWrite.h"
#include "FeatureData.h"
#include "distance.h"
#include <string>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;
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
  Mat image;
  bool train_mode = false;
  vector<FeatureData> trainingData;
  string label;
  char key;
  string csvFileName = "trainingdata.csv";
  int maxRegions = 5;
  vector<Vec3b> colors;

  // keep track of colors here
  for (int i = 0; i < maxRegions; i++)
  {
    Vec3b newColor(rand() & 255, rand() & 255, rand() & 255);
    colors.push_back(newColor);
  }

  // read existing training data first
  read(trainingData, csvFileName);

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

    // initialize variables
    Mat original = image;
    Mat stats, labels, colour_labels;
    int num_labels;
    double percentFilled, heightWidthRatio;

    // show normal video
    if (!train_mode)
    {
      namedWindow("Live Video", WINDOW_NORMAL);
      imshow("Live Video", image);
    }

    // threshold
    threshold(image);

    // show thresholded video
    if (!train_mode)
    {
      namedWindow("Thresholded", WINDOW_NORMAL);
      imshow("Thresholded", image);
    }

    // clean up
    cleanup(image);

    // show cleaned up video
    if (!train_mode)
    {
      namedWindow("Cleaned", WINDOW_NORMAL);
      imshow("Cleaned", image);
    }

    // segment into colored regions
    priority_queue<pair<double, int>> minDistances;
    labels = labelConnectedComponents(image, 3000, stats, num_labels, maxRegions, minDistances);
    colour_labels = colorlabelConnectedComponents(labels, colors, minDistances);

    // show segmented video
    if (!train_mode)
    {
      namedWindow("Coloured", WINDOW_NORMAL);
      imshow("Coloured", colour_labels);
    }

    // compute features of each region in minDistances
    priority_queue<pair<double, int>> tempDistances = minDistances;

    // calculate standard deviations of feature data
    vector<double> stdev;
    vector<double> f1_data;
    vector<double> f2_data;

    // get all f1 and f2 data into vector
    for (int i = 0; i < trainingData.size(); i++)
    {
      f1_data.push_back(trainingData.at(i).f1);
      f2_data.push_back(trainingData.at(i).f2);
    }

    // store standard deviations
    stdev.push_back(calculate_stdev(f1_data));
    stdev.push_back(calculate_stdev(f2_data));

    // if not in training mode show all objects
    if (!train_mode)
    {
      while (!tempDistances.empty())
      {
        auto top = tempDistances.top();

        computeFeatures(labels, top.second, percentFilled, heightWidthRatio, image, original); // the bounding boxes and features will be shown on the original image

        FeatureData currFD = {"current", percentFilled, heightWidthRatio};

        // compare features of current region to all training data and see what the closest label is using euclidean distance
        double thresholdClosestLabel = 3;
        auto start1 = high_resolution_clock::now();
        string itemName = closestLabelScaledEuclidean(currFD, trainingData, thresholdClosestLabel, stdev);
        auto stop1 = high_resolution_clock::now();
        auto duration1 = duration_cast<microseconds>(stop1 - start1);
        cout << "Time taken for closestLabelScaledEuclidean: " << duration1.count() << " microseconds" << endl;
        cout << "Closest label found with nearest neighbor (euclidean distance) - " << itemName << endl;

        // compare features of current region to all training data and see what the closest label is using cosine similarity
        double thresholdCosine = 0;
        auto start2 = high_resolution_clock::now();
        string itemName_cos = closestLabelCosineSimilarity(currFD, trainingData, thresholdCosine);
        auto stop2 = high_resolution_clock::now();
        auto duration2 = duration_cast<microseconds>(stop2 - start2);
        cout << "Time taken for closestLabelCosineSimilarity: " << duration2.count() / 1e6 << " microseconds" << endl;
        cout << "Closest label found with nearest neighbor (cosine similarity) - " << itemName_cos << endl;

        if (trainingData.size() > 0)
        {
          // use KNN
          auto start3 = high_resolution_clock::now();
          string itemName_KNN = kNearestLabels(currFD, trainingData, stdev, 3).at(0); // get closest match
          auto stop3 = high_resolution_clock::now();
          auto duration3 = duration_cast<microseconds>(stop3 - start3);
          cout << "Time taken for kNearestLabels: " << duration3.count() << " microseconds" << endl;
          cout << "Closest label found with KNN - " << itemName_KNN << endl;
        }

        // create label at center of region
        int centroidX = stats.at<int>(top.second, CC_STAT_LEFT) + stats.at<int>(top.second, CC_STAT_WIDTH) / 2;
        int centroidY = stats.at<int>(top.second, CC_STAT_TOP) + stats.at<int>(top.second, CC_STAT_HEIGHT) / 2;

        stringstream ss;
        ss << "Label: " << itemName;
        putText(original, ss.str(), Point(centroidX, centroidY), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 6);

        tempDistances.pop();
      }
    }
    else
    // if in training mode, show just most centered object
    {
      pair<double, int> lastElement;
      while (!tempDistances.empty())
      {
        lastElement = tempDistances.top();
        tempDistances.pop();
      }
      computeFeatures(labels, lastElement.second, percentFilled, heightWidthRatio, image, original); // the bounding boxes and features will be shown on the original image
    }

    // show features video
    namedWindow("Features", WINDOW_NORMAL);
    imshow("Features", original);

    // see if there is a waiting keystroke
    char keyPressed = waitKey(1);
    if (keyPressed == 'q')
    {
      break;
    }

    switch (keyPressed)
    {
    case 'T': // T - Training Mode
      train_mode = !train_mode;
      if (train_mode)
      {
        cout << "Training Mode activated." << endl;
      }
      else
      {
        cout << "Object Detection Mode activated." << endl;
      }
      break;
    case 'N': // N - Take input and save as feature vector
      if (!train_mode)
      {
        cout << "Training Mode should be activated." << endl;
      }
      else
      {
        cout << "Enter the label." << endl;
        cin >> label; // get label name
        // add feature vector to training data
        FeatureData temp = {label, percentFilled, heightWidthRatio};
        trainingData.push_back(temp);
      }
      break;
    }
  }
  delete capdev;

  // write training data
  write(trainingData, csvFileName);

  return (0);
}