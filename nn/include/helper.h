#ifndef __HELPER__H
#define __HELPER__H

#include <string>
#include <sstream>
#include <vector>
#include <cassert>

std::vector<int> parseInts(const std::string &s, char delim);
std::vector<double> parseDoubles(const std::string &s, char delim);

int getIndexWithMaxValue(std::vector<double> v);

double computeError(const std::vector<double> v1,
                    const std::vector<double> v2);

#endif

