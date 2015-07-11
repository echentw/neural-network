#include <iostream>
#include <fstream>
#include <cassert>

#include "neural_network.h"
#include "color_print.h"

using namespace std;

void TestComputeOutput() {
  // TODO: test me properly
  int n_outputs = 10;
  NeuralNetwork network(n_outputs);
  network.addInputLayer(784);
  network.addHiddenLayer(30);
  network.addOutputLayer();

  vector<double> input(784, 1.0);
  vector<double> output = network.computeOutput(input);

  cout << output.size() << endl;
  for (int i = 0; i < n_outputs; ++i) {
    cout << output[i] << endl;
  }
}

void TestBackpropagate() {
  // TODO: implement me!
}

void TestSimple() {
  double epsilon = 0.1;

  // initializing input data
  vector<double> input_data1(3, 1.0);
  vector<double> input_data2(3, 0.5);
  vector<double> input_data3(3, 0.0);
  vector<vector<double> > inputs;
  inputs.push_back(input_data1);
  inputs.push_back(input_data2);
  inputs.push_back(input_data3);

  // initializing labels
  vector<double> output_data1(2, 1.0);
  vector<double> output_data2(2, 0.5);
  vector<double> output_data3(2, 0.0);
  vector<vector<double> > outputs;
  outputs.push_back(output_data1);
  outputs.push_back(output_data2);
  outputs.push_back(output_data3);

  // creating a neural network instance
  NeuralNetwork network(2);
  network.addInputLayer(3);
  network.addHiddenLayer(10);
  network.addHiddenLayer(5);
  network.addOutputLayer();
  network.setThreshold(1e-4);

  // train and save the data
  network.train(inputs, outputs);
  network.save("config.txt");

  // check the network has been trained
  for (int i = 0; i < 3; ++i) {
    vector<double> output = network.computeOutput(inputs[i]);
    assert(output.size() == 2);
    assert(output[0] < outputs[i][0] + epsilon);
    assert(output[0] > outputs[i][0] - epsilon);
    assert(output[1] < outputs[i][1] + epsilon);
    assert(output[1] > outputs[i][1] - epsilon);
  }

  printPass("TestSimple()");
}

void TestLoading() {
  NeuralNetwork network(10);
  network.load("config.txt");
  network.save("config2.txt");

  ifstream file1("config.txt"), file2("config2.txt");
  string line1, line2;
  while (getline(file1, line1) && getline(file2, line2)) {
     assert(line1 == line2);
  }

  file1.close();
  file2.close();

  printPass("TestLoad()");
}


int main() {
//  TestComputeOutput(); // TODO: write proper test
//  TestBackpropagate(); // TODO: write proper test
  TestSimple();
  TestLoading();
}

