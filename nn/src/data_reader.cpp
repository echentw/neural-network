#include "data_reader.h"

DataReader::DataReader(std::string input_filepath) {
  this->input_filepath = input_filepath;
  this->label_filepath = ""; 
}

DataReader::DataReader(std::string input_filepath,
                       std::string label_filepath) {
  this->input_filepath = input_filepath;
  this->label_filepath = label_filepath;
}

std::vector<std::vector<double> > DataReader::convertInputData(int dim,
                                                               double scale) {
  std::ifstream file(input_filepath);
  std::string line;
  std::vector<std::vector<double> > input_data;
  while (std::getline(file, line)) {
    std::vector<double> data = parseDoubles(line, ' ');
    assert(data.size() == dim);
    for (int i = 0; i < data.size(); ++i) {
      data[i] *= scale;
    }
    input_data.push_back(data);
  }
  return input_data;
}

std::vector<int> DataReader::convertLabels() {
  std::ifstream file(label_filepath);
  std::string line;
  std::vector<int> labels;
  while (std::getline(file, line)) {
    std::vector<double> data = parseDoubles(line, ' ');
    assert(data.size() == 1);
    labels.push_back(data[0]);
  }
  return labels;
}

