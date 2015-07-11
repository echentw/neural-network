#include "output_node.h"

OutputNode::OutputNode() {
  this->input = 0.0;
  this->output = 0.0;
}

double OutputNode::getInput() const {
  return this->input;
}

double OutputNode::getOutput() const {
  return this->output;
}

void OutputNode::receiveInput(double input) {
  this->input = input;
  this->output = input;
}

