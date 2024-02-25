/*
	Samuel Lee
	Anjith Prakash Chathan Kandy
	2/21/24
	This file contains functions to read and write csv files.
*/

#include <iostream>
#include <fstream>
#include <string>
#include "FeatureData.h"

using namespace std;

/**
 * This function reads a csv file into a vector of FeatureData
 * data - all FeatureData
 * fileName - name of the csv file
*/
void read(vector<FeatureData> &data, string &fileName) {
	ifstream inFile(fileName); // open file

	// error opening file
	if(!inFile) {
		cerr << "Error Unable to open the file!" << endl;
		return ;
	}

	// read one line into temp until no next line exists
	FeatureData temp;
	while (inFile >> temp.label >> temp.f1 >> temp.f2) {
		data.push_back(temp);
	}

	inFile.close(); // close file after reading
}

/**
 * This function writes a csv file from a vector of FeatureData
 * data - all FeatureData
 * fileName - name of the csv file
*/
void write(vector<FeatureData> &data, string &fileName) {
	ofstream outFile(fileName); // open file

	// error opening file
	if(!outFile) {
		cerr << "Error: Unable to open the file!" << endl;
		return;
	}

	// for each data point, add to outfile
	for(const auto& item : data) {
		outFile << item.label << " " << item.f1 << " " << item.f2 << endl;
	}

	outFile.close(); // close file after writing
}