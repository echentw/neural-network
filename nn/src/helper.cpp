#include "helper.h"

std::vector<int> parseInts(const std::string &s, char delim) {
  std::vector<int> elems;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back( std::atoi(item.c_str()) );
  }
  return elems;
}

std::vector<double> parseDoubles(const std::string &s, char delim) {
  std::vector<double> elems;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back( std::atof(item.c_str()) );
  }
  return elems;
}

int getIndexWithMaxValue(std::vector<double> v) {
  if (v.size() == 0) {
    return -1;
  }
  int id = 0;
  double max_value = v[0];
  for (int i = 0; i < v.size(); ++i) {
    if (v[i] > v[id]) {
      max_value = v[i];
      id = i;
    }
  }
  return id;
}

double computeError(const std::vector<double> v1,
                    const std::vector<double> v2) {
  assert(v1.size() == v2.size());
  double sum = 0.0;
  for (int i = 0; i < v2.size(); ++i) {
    sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
  }
  return sum;
}

