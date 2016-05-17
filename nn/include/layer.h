#ifndef __LAYER__H
#define __LAYER__H

#include <vector>

#include "node.h"
#include "input_node.h"
#include "hidden_node.h"
#include "output_node.h"

class Layer {
 protected:
  int size;

 public:
  int getSize() const {
    return size;
  }
  virtual double getInput(int i) const = 0;
  virtual double getOutput(int i) const = 0;
  virtual std::vector<double> getInput() const = 0;
  virtual std::vector<double> getOutput() const = 0;
  virtual void receiveInput(const std::vector<double>& input) = 0;

  virtual double getPartialDerivative(int input_id) const = 0;
};

#endif

