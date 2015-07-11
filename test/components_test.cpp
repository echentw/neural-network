#include <iostream>

#include "input_layer.h"
#include "hidden_layer.h"
#include "output_layer.h"
#include "weight_matrix.h"
#include "input_node.h"
#include "hidden_node.h"
#include "output_node.h"

#include "color_print.h"

void TestInputNode() {
  InputNode input_node1;
  input_node1.receiveInput(2.0);
  assert(input_node1.getInput() == 2.0);
  assert(input_node1.getOutput() == 2.0);

  InputNode input_node2;
  input_node2.receiveInput(-2.0);
  assert(input_node2.getInput() == -2.0);
  assert(input_node2.getOutput() == -2.0);

  printPass("TestInputNode()");
}

void TestHiddenNode() {
  HiddenNode hidden_node1;
  hidden_node1.receiveInput(3.0);
  assert(hidden_node1.getInput() == 3.0);
  assert(hidden_node1.getOutput() == 3.0);

  HiddenNode hidden_node2;
  hidden_node2.receiveInput(-4.0);
  assert(hidden_node2.getInput() == -4.0);
  assert(hidden_node2.getOutput() == 0.0);

  printPass("TestHiddenNode()");
}

void TestOutputNode() {
  OutputNode output_node1;
  output_node1.receiveInput(3.0);
  assert(output_node1.getInput() == 3.0);
  assert(output_node1.getOutput() == 3.0);

  OutputNode output_node2;
  output_node2.receiveInput(-4.0);
  assert(output_node2.getInput() == -4.0);
  assert(output_node2.getOutput() == -4.0);

  printPass("TestOutputNode()");
}

void TestInputLayer() {
  std::vector<double> values;
  values.push_back(-1.0);
  values.push_back(0.0);
  values.push_back(1.0);

  InputLayer layer(values.size());
  layer.receiveInput(values);

  assert(layer.getSize() == 3);

  std::vector<double> input = layer.getInput();
  std::vector<double> output = layer.getOutput();

  assert(input[0] == -1.0);
  assert(input[1] == 0.0);
  assert(input[2] == 1.0);

  assert(output[0] == -1.0);
  assert(output[1] == 0.0);
  assert(output[2] == 1.0);

  printPass("TestInputLayer()");
}

void TestHiddenLayer_Forward() {
  std::vector<double> values;
  values.push_back(-1.0);
  values.push_back(0.0);
  values.push_back(1.0);

  HiddenLayer layer(values.size());
  layer.receiveInput(values);

  assert(layer.getSize() == 3);

  std::vector<double> input = layer.getInput();
  std::vector<double> output = layer.getOutput();

  assert(input[0] == -1.0);
  assert(input[1] == 0.0);
  assert(input[2] == 1.0);

  assert(output[0] == 0.0);
  assert(output[1] == 0.0);
  assert(output[2] == 1.0);

  printPass("TestHiddenLayer_Forward()");
}

void TestOutputLayer_Forward() {
  std::vector<double> values;
  values.push_back(-1.0);
  values.push_back(0.0);
  values.push_back(1.0);

  OutputLayer layer(values.size());
  layer.receiveInput(values);

  assert(layer.getSize() == 3);

  std::vector<double> input = layer.getInput();
  std::vector<double> output = layer.getOutput();

  assert(input[0] == -1.0);
  assert(input[1] == 0.0);
  assert(input[2] == 1.0);

  assert(output[0] == -1.0);
  assert(output[1] == 0.0);
  assert(output[2] == 1.0);

  printPass("TestOutputLayer_Forward()");
}

void TestWeightMatrix_Forward() {
  int n_outputs = 2;
  InputLayer input_layer(3);
  OutputLayer output_layer(n_outputs);

  WeightMatrix weights(input_layer, output_layer);
  weights.set(0, 0, 0.5);
  weights.set(1, 0, -2.0);
  weights.set(2, 0, 1.5);
  weights.set(0, 1, 1.0);
  weights.set(1, 1, 0.7);
  weights.set(2, 1, -1.0);
  weights.setBias(0, 0.8);
  weights.setBias(1, -0.3);

  std::vector<double> inputs;
  inputs.push_back(-2.0);
  inputs.push_back(1.0);
  inputs.push_back(3.0);
  input_layer.receiveInput(inputs);

  std::vector<double> transition = weights.fire(input_layer);

  assert(transition.size() == 2);
  assert(transition[0] == 2.3);
  assert(transition[1] == -4.6);

  output_layer.receiveInput(transition);

  assert(output_layer.getInput(0) == 2.3);
  assert(output_layer.getInput(1) == -4.6);

  assert(output_layer.getOutput(0) == 2.3);
  assert(output_layer.getOutput(1) == -4.6);

  printPass("TestWeightMatrix_Forward()");
}

void TestOutputLayer_Backward() {
  double epsilon = 1e-3;

  std::vector<double> values;
  values.push_back(1.0);
  values.push_back(1.0);
  values.push_back(-1.0);

  std::vector<double> output;
  output.push_back(-2.2);
  output.push_back(3.4);
  output.push_back(5.5);

  OutputLayer layer(values.size());
  layer.receiveInput(values);

  assert(layer.getSize() == 3);

  layer.computeGradient(output);
  
  assert(layer.getPartialDerivative(0) < 3.2/3 + epsilon &&
         layer.getPartialDerivative(0) > 3.2/3 - epsilon);
  assert(layer.getPartialDerivative(1) < -2.4/3 + epsilon &&
         layer.getPartialDerivative(1) > -2.4/3 - epsilon);
  assert(layer.getPartialDerivative(2) < -6.5/3 + epsilon &&
         layer.getPartialDerivative(2) > -6.5/3 - epsilon);

  printPass("TestOutputLayer_Backward()");
}

void TestWeightMatrix_Backward() {
  double epsilon = 1e-3;

  int n_outputs = 2;
  int n_inputs = 3;
  InputLayer input_layer(n_inputs);
  OutputLayer output_layer(n_outputs);

  WeightMatrix weights(input_layer, output_layer);
  weights.set(0, 0, 0.5);
  weights.set(1, 0, -2.0);
  weights.set(2, 0, 1.5);
  weights.set(0, 1, 1.0);
  weights.set(1, 1, 0.7);
  weights.set(2, 1, -1.0);
  weights.setBias(0, 0.8);
  weights.setBias(1, -0.3);

  std::vector<double> inputs;
  inputs.push_back(-2.0);
  inputs.push_back(1.0);
  inputs.push_back(3.0);
  input_layer.receiveInput(inputs);

  std::vector<double> transition = weights.fire(input_layer);
  output_layer.receiveInput(transition);

  assert(output_layer.getInput(0) == 2.3);
  assert(output_layer.getInput(1) == -4.6);

  // backpropagation step
  std::vector<double> actual_outputs;
  actual_outputs.push_back(1.0);
  actual_outputs.push_back(1.0);

  output_layer.computeGradient(actual_outputs);
  weights.computeGradient(input_layer, output_layer);

  assert(weights.getPartialDerivative(0, 0) > -1.3 - epsilon &&
         weights.getPartialDerivative(0, 0) < -1.3 + epsilon);
  assert(weights.getPartialDerivative(1, 0) > 0.65 - epsilon &&
         weights.getPartialDerivative(1, 0) < 0.65 + epsilon);
  assert(weights.getPartialDerivative(2, 0) > 1.95 - epsilon &&
         weights.getPartialDerivative(2, 0) < 1.95 + epsilon);

  assert(weights.getPartialDerivative(0, 1) > 5.6 - epsilon &&
         weights.getPartialDerivative(0, 1) < 5.6 + epsilon);
  assert(weights.getPartialDerivative(1, 1) > -2.8 - epsilon &&
         weights.getPartialDerivative(1, 1) < -2.8 + epsilon);
  assert(weights.getPartialDerivative(2, 1) > -8.4 - epsilon &&
         weights.getPartialDerivative(2, 1) < -8.4 + epsilon);

  assert(weights.getBiasPartialDerivative(0) < 0.65 + epsilon &&
         weights.getBiasPartialDerivative(0) > 0.65 - epsilon);
  assert(weights.getBiasPartialDerivative(1) < -2.8 + epsilon &&
         weights.getBiasPartialDerivative(1) > -2.8 - epsilon);

  printPass("TestWeightMatrix_Backward()");
}

void TestHiddenLayer_Backward() {
  double epsilon = 1e-3;

  int n_outputs = 2;
  int n_inputs = 3;
  HiddenLayer hidden_layer(n_inputs);
  OutputLayer output_layer(n_outputs);

  std::vector<double> hidden_values;
  hidden_values.push_back(-2.0);
  hidden_values.push_back(1.0);
  hidden_values.push_back(3.0);

  WeightMatrix weights(hidden_layer, output_layer);
  weights.set(0, 0, 0.5);
  weights.set(1, 0, -2.0);
  weights.set(2, 0, 1.5);
  weights.set(0, 1, 1.0);
  weights.set(1, 1, 0.7);
  weights.set(2, 1, -1.0);
  weights.setBias(0, 0.8);
  weights.setBias(1, -0.3);

  hidden_layer.receiveInput(hidden_values);
  std::vector<double> transition = weights.fire(hidden_layer);
  output_layer.receiveInput(transition);

  assert(output_layer.getInput(0) == 3.3);
  assert(output_layer.getInput(1) == -2.6);

  // backpropagation step
  std::vector<double> actual_outputs;
  actual_outputs.push_back(1.0);
  actual_outputs.push_back(1.0);

  output_layer.computeGradient(actual_outputs);
  weights.computeGradient(hidden_layer, output_layer);
  hidden_layer.computeGradient(weights, output_layer);

  assert(hidden_layer.getPartialDerivative(0) < 0.0 + epsilon &&
         hidden_layer.getPartialDerivative(0) > 0.0 - epsilon);
  assert(hidden_layer.getPartialDerivative(1) < -3.56 + epsilon &&
         hidden_layer.getPartialDerivative(1) > -3.56 - epsilon);
  assert(hidden_layer.getPartialDerivative(2) < 3.525 + epsilon &&
         hidden_layer.getPartialDerivative(2) > 3.525 - epsilon);

  printPass("TestHiddenLayer_Backward()");
}

int main() {
  // testing the nodes
  TestInputNode();
  TestHiddenNode();
  TestOutputNode();

  // forward propagation things
  TestInputLayer();
  TestHiddenLayer_Forward();
  TestOutputLayer_Forward();
  TestWeightMatrix_Forward();

  // backward propagation things
  TestOutputLayer_Backward();
  TestWeightMatrix_Backward();
  TestHiddenLayer_Backward();
}

