#ifndef __DATA_READER__H
#define __DATA_READER__H

#include <string>
#include <vector>
#include <fstream>
#include <cassert>

#include "helper.h"

class DataReader {
 private:
  std::string input_filepath;
  std::string label_filepath;

 public:
  DataReader(std::string input_filepath);
  DataReader(std::string input_filepath, std::string label_filepath);

  std::vector<std::vector<double> > convertInputData(int dim,
                                                     double scale = 1.0);
  std::vector<int> convertLabels();
};

#endif

