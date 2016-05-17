#ifndef __NODE__H
#define __NODE__H

#include <algorithm>

class Node {
 protected:
  double input;
  double output;

 public:
  virtual double getInput() const = 0;
  virtual double getOutput() const = 0;
  virtual void receiveInput(double input) = 0;
};

#endif

