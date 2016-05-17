#include <iostream>

#include "neural_network.h"
#include "data_reader.h"

void TrainMNIST(std::string rootpath) {
  std::string input_filepath = rootpath + "train_images.txt";
  std::string label_filepath = rootpath + "train_labels.txt";

  int dim = 28 * 28;

  DataReader reader(input_filepath, label_filepath);

  std::cout << "Reading input data...\n"; std::vector<std::vector<double> > input_data =
      reader.convertInputData(dim, 1/256.0);

  std::cout << "Reading labels...\n";
  std::vector<int> raw_labels = reader.convertLabels();

  int num_outputs = 10;

  std::cout << "Converting labels...\n";
  std::vector<std::vector<double> > labels;
  for (int i = 0; i < raw_labels.size(); ++i) {
    std::vector<double> label(num_outputs, 0.0);
    label[ raw_labels[i] ] = 10.0;
    labels.push_back(label);
  }

  NeuralNetwork network(num_outputs);
  network.addInputLayer(dim);
  network.addHiddenLayer(30);
  network.addOutputLayer();

  std::cout << "Loading network..." << std::endl;
  network.load("mnist_config4.txt");

  network.setBatchSize(500);
  network.setStepSize(1e-3);

  std::cout << "Training network..." << std::endl;
  network.train(input_data, labels, 1, "mnist_config5.txt");
}

void SeeTrainingAccuracy(std::string rootpath) {
  std::string input_filepath = rootpath + "train_images.txt";
  std::string label_filepath = rootpath + "train_labels.txt";

  int dim = 28 * 28;

  DataReader reader(input_filepath, label_filepath);

  std::cout << "Reading input data...\n";
  std::vector<std::vector<double> > input_data =
      reader.convertInputData(dim, 1/256.0);

  std::cout << "Reading labels...\n";
  std::vector<int> raw_labels = reader.convertLabels();

  int num_outputs = 10;

  std::cout << "Converting labels...\n";
  std::vector<std::vector<double> > labels;
  for (int i = 0; i < raw_labels.size(); ++i) {
    std::vector<double> label(num_outputs, 0.0);
    label[ raw_labels[i] ] = 1.0;
    labels.push_back(label);
  }

  NeuralNetwork network(num_outputs);
  network.load("mnist_config4.txt");

  std::cout << "Testing..." << std::endl;
  network.test(input_data, labels);

}

void TestMNIST(std::string rootpath) {
  std::string input_filepath = rootpath + "test_images.txt";
  std::string label_filepath = rootpath + "test_labels.txt";

  int dim = 28 * 28;

  DataReader reader(input_filepath, label_filepath);

  std::cout << "Reading input data...\n";
  std::vector<std::vector<double> > input_data =
      reader.convertInputData(dim, 1/256.0);

  std::cout << "Reading labels...\n";
  std::vector<int> raw_labels = reader.convertLabels();

  int num_outputs = 10;

  std::cout << "Converting labels...\n";
  std::vector<std::vector<double> > labels;
  for (int i = 0; i < raw_labels.size(); ++i) {
    std::vector<double> label(num_outputs, 0.0);
    label[ raw_labels[i] ] = 1.0;
    labels.push_back(label);
  }

  NeuralNetwork network(num_outputs);
  network.load("mnist_config4.txt");

  network.test(input_data, labels);
}

int main() {
  std::string rootpath =
      "/home/echentw/Documents/projects/github/neural-network/mnist/txt_data/";
  TrainMNIST(rootpath);
  // SeeTrainingAccuracy(rootpath);
  // TestMNIST(rootpath);
  return 0;
}

