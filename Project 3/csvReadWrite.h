/*
	Samuel Lee
	Anjith Prakash Chathan Kandy
	2/21/24
	Header file to support CSV operations
*/

#ifndef CSVREADWRITE_H
#define CSVREADWRITE_H

#include <vector>
#include <string>
#include "FeatureData.h"

// function declarations
void read(std::vector<FeatureData> &data, std::string &fileName);
void write(std::vector<FeatureData> &data, std::string &fileName);

#endif // CSVREADWRITE_H