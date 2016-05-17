#ifndef __INPUT_LAYER__H
#define __INPUT_LAYER__H

#include "layer.h"
#include "input_node.h"

class InputLayer : public Layer {
 private:
  std::vector<InputNode> node_list;

 public:
  InputLayer(int size);
  double getInput(int i) const;
  double getOutput(int i) const;
  std::vector<double> getInput() const;
  std::vector<double> getOutput() const;
  void receiveInput(const std::vector<double>& input);

  // only here because needs a concrete implementation
  // returns 0
  double getPartialDerivative(int input_id) const;
};

#endif

