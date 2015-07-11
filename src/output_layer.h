#ifndef __OUTPUT_LAYER__H
#define __OUTPUT_LAYER__H

#include "layer.h"
#include "output_node.h"

#include "boost/multi_array.hpp"

class OutputLayer : public Layer {
 private:
  std::vector<OutputNode> node_list;

  // gradients[i] = partial derivative of cost w.r.t. input i
  boost::multi_array<double, 1> gradients;

 public:
  OutputLayer(int size);
  void adjust(int new_size);
  double getInput(int i) const;
  double getOutput(int i) const;
  std::vector<double> getInput() const;
  std::vector<double> getOutput() const;
  void receiveInput(const std::vector<double>& input);

  void computeGradient(const std::vector<double>& output);

  // returns the partial derivative of cost
  // w.r.t. the input #input_id of this layer
  double getPartialDerivative(int input_id) const;
};

#endif

