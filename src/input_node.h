#ifndef __INPUT_NODE__H
#define __INPUT_NODE__H

#include "node.h"

class InputNode : public Node {
 public:
  InputNode();
  double getInput() const;
  double getOutput() const;
  void receiveInput(double input);
};

#endif

