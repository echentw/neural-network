#ifndef __HIDDEN_LAYER__H
#define __HIDDEN_LAYER__H

#include "layer.h"
#include "hidden_node.h"
#include "weight_matrix.h"

#include "boost/multi_array.hpp"

class HiddenLayer : public Layer {
 private:
  std::vector<HiddenNode> node_list;
  boost::multi_array<double, 1> gradients;

 public:
  HiddenLayer(int size);
  double getInput(int i) const;
  double getOutput(int i) const;
  std::vector<double> getInput() const;
  std::vector<double> getOutput() const;
  void receiveInput(const std::vector<double>& input);
  void computeGradient(const WeightMatrix& weights,
                       const Layer& next_layer);

  // returns the partial derivative of cost
  // w.r.t. the input #input_id of this layer
  double getPartialDerivative(int input_id) const;
};

#endif

