#include "input_layer.h"

InputLayer::InputLayer(int size) {
  this->size = size;
  this->node_list.resize(size);
}

double InputLayer::getInput(int i) const {
  return this->node_list[i].getInput();
}

double InputLayer::getOutput(int i) const {
  return this->node_list[i].getOutput();
}

std::vector<double> InputLayer::getInput() const {
  std::vector<double> input(this->size, 0.0);
  for (int i = 0; i < this->size; ++i) {
    input[i] = this->node_list[i].getInput();
  }
  return input;
}

std::vector<double> InputLayer::getOutput() const {
  std::vector<double> output(this->size, 0.0);
  for (int i = 0; i < this->size; ++i) {
    output[i] = this->node_list[i].getOutput();
  }
  return output;
}

void InputLayer::receiveInput(const std::vector<double>& input) {
  for (int i = 0; i < this->size; ++i) {
    this->node_list[i].receiveInput(input[i]);
  }
}

double InputLayer::getPartialDerivative(int input_id) const {
  return 0.0;
}

