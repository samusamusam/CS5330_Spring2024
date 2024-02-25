/*
	Samuel Lee
	Anjith Prakash Chathan Kandy
	2/21/24
	This file contains functions to threshold an image and convert it to a binary image.
*/

#include <cstdio>
#include <cstring>
#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

/**
 * This function applies the threshold to the image
 * src - source image
 * thresholdValue - value to threshold by
 */
void thresholdImage(Mat &src, int thresholdValue)
{
	cvtColor(src, src, COLOR_BGR2GRAY); // convert to grayscale

	for (int i = 0; i < src.rows; i++)
	{
		uchar *dptr = src.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++)
		{
			uchar pixelValue = dptr[j];
			if (pixelValue < thresholdValue)
			{
				dptr[j] = 255; // set to white
			}
			else
			{
				dptr[j] = 0; // set to black
			}
		}
	}
}

/**
 * This function returns a thresholded image
 * src - source image
 */
void threshold(Mat &src)
{
	// initialize variables
	Mat temp;

	GaussianBlur(src, src, Size(5, 5), 0); // apply Gaussian blur

	// converts to HSV for darkening the image if saturation > 150
	cvtColor(src, src, COLOR_BGR2HSV);
	for (int i = 0; i < src.rows; i++)
	{
		Vec3b *dptr = src.ptr<Vec3b>(i);
		for (int j = 0; j < src.cols; j++)
		{
			Vec3b pixel = dptr[j];
			int saturation = pixel[1];
			if (saturation > 150)
			{
				pixel[2] = round(pixel[2] * 0.5);
			}
			dptr[j] = pixel;
		}
	}

	// conver HSV to BGR
	cvtColor(src, src, COLOR_HSV2BGR);

	temp = src.clone(); // store src into temp for later use

	// converting to CV_32F for kmeans
	src.convertTo(src, CV_32F);

	// run k-means
	int clusterCount = 2;
	Mat labels;
	Mat centers;

	kmeans(src, clusterCount, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

	// calculate the average intensity
	float center1 = (centers.at<float>(0, 0) + centers.at<float>(0, 1) + centers.at<float>(0, 2)) / 3.0f;
	float center2 = (centers.at<float>(1, 0) + centers.at<float>(1, 1) + centers.at<float>(1, 2)) / 3.0f;
	int thresholdValue = static_cast<int>((center1 + center2) / 2.0f);

	temp.copyTo(src);

	// release allocation of memory to temp
	temp.release();
	temp = Mat();

	// create thresholded image
	thresholdImage(src, thresholdValue);
}