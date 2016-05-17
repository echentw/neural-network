#include "output_layer.h"

OutputLayer::OutputLayer(int size) {
  this->size = size;
  this->node_list.resize(size);

  boost::multi_array<double, 1>::extent_gen extents;
  this->gradients.resize(extents[size]);
}

void OutputLayer::adjust(int new_size) {
  this->size = new_size;
  this->node_list.resize(new_size);

  boost::multi_array<double, 1>::extent_gen extents;
  this->gradients.resize(extents[new_size]);
}

double OutputLayer::getInput(int i) const {
  return this->node_list[i].getInput();
}

double OutputLayer::getOutput(int i) const {
  return this->node_list[i].getOutput();
}

std::vector<double> OutputLayer::getInput() const {
  std::vector<double> input(this->size, 0.0);
  for (int i = 0; i < this->size; ++i) {
    input[i] = this->node_list[i].getInput();
  }
  return input;
}

std::vector<double> OutputLayer::getOutput() const {
  std::vector<double> output(this->size, 0.0);
  for (int i = 0; i < this->size; ++i) {
    output[i] = this->node_list[i].getOutput();
  }
  return output;
}

void OutputLayer::receiveInput(const std::vector<double>& input) {
  for (int i = 0; i < this->size; ++i) {
    this->node_list[i].receiveInput(input[i]);
  }
}

void OutputLayer::computeGradient(const std::vector<double>& output) {
  for (int i = 0; i < this->size; ++i) {
    this->gradients[i] = (node_list[i].getOutput() - output[i]) / size;
  }
}

double OutputLayer::getPartialDerivative(int input_id) const {
  return this->gradients[input_id];
}

