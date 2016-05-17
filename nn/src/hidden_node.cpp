#include "hidden_node.h"

HiddenNode::HiddenNode() {
  this->input = 0.0;
  this->output = 0.0;
}

double HiddenNode::getInput() const {
  return this->input;
}

double HiddenNode::getOutput() const {
  return this->output;
}

void HiddenNode::receiveInput(double input) {
  this->input = input;
  this->output = std::max<double>(0.0, input);
}

