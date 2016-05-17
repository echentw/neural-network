#include "hidden_layer.h"

HiddenLayer::HiddenLayer(int size) {
  this->size = size;
  this->node_list.resize(size);

  boost::multi_array<double, 1>::extent_gen extents;
  this->gradients.resize(extents[size]);
}

double HiddenLayer::getInput(int i) const {
  return this->node_list[i].getInput();
}

double HiddenLayer::getOutput(int i) const {
  return this->node_list[i].getOutput();
}

std::vector<double> HiddenLayer::getInput() const {
  std::vector<double> input(this->size, 0.0);
  for (int i = 0; i < this->size; ++i) {
    input[i] = this->node_list[i].getInput();
  }
  return input;
}

std::vector<double> HiddenLayer::getOutput() const {
  std::vector<double> output(this->size, 0.0);
  for (int i = 0; i < this->size; ++i) {
    output[i] = this->node_list[i].getOutput();
  }
  return output;
}

void HiddenLayer::receiveInput(const std::vector<double>& input) {
  for (int i = 0; i < this->size; ++i) {
    this->node_list[i].receiveInput(input[i]);
  }
}

void HiddenLayer::computeGradient(const WeightMatrix& weights,
                                  const Layer& next_layer) {
  for (int i = 0; i < this->size; ++i) {
    this->gradients[i] = 0.0;
    if (this->getInput(i) > 0.0) {
      for (int j = 0; j < next_layer.getSize(); ++j) {
        this->gradients[i] +=
            next_layer.getPartialDerivative(j) * weights.get(i, j);
      }
    }
  }
}

double HiddenLayer::getPartialDerivative(int input_id) const {
  return this->gradients[input_id];
}

