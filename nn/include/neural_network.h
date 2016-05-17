#ifndef __NEURAL_NETWORK__H
#define __NEURAL_NETWORK__H

#include <string>
#include <fstream>
#include <cassert>

#include "layer.h"
#include "input_layer.h"
#include "hidden_layer.h"
#include "output_layer.h"
#include "weight_matrix.h"
#include "helper.h"

class NeuralNetwork {
 private:
  int n_outputs;
  InputLayer input_layer;
  std::vector<HiddenLayer> hidden_layers;
  OutputLayer output_layer;
  std::vector<WeightMatrix> weight_matrices;

  double batch_size = 10000;
  double step_size = 1e-3;
  double threshold = 1e-3;
  int iteration = 0;

 public:
  NeuralNetwork(int n_outputs);
  void addInputLayer(int size);
  void addHiddenLayer(int size);
  void addOutputLayer();

  double getIterationNumber() const;
  double getBatchSize() const;
  double getStepSize() const;
  double getThreshold() const;

  void setStepSize(double step_size);
  void setBatchSize(double batch_size);
  void setThreshold(double threshold);

  std::vector<double> computeOutput(std::vector<double> input);

  void backpropagate(std::vector<double> correct_output);

  void train(std::vector<std::vector<double> > inputs,
             std::vector<std::vector<double> > labels,
             int save_period = -1,
             std::string save_filename = "");

  void test(std::vector<std::vector<double> > inputs,
            std::vector<std::vector<double> > labels);

  void save(std::string filename) const;

  void load(std::string filename);
};

#endif

