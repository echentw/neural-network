#include "input_node.h"

InputNode::InputNode() {
  this->input = 0.0;
  this->output = 0.0;
}

double InputNode::getInput() const {
  return this->input;
}

double InputNode::getOutput() const {
  return this->output;
}

void InputNode::receiveInput(double input) {
  this->input = input;
  this->output = input;
}

