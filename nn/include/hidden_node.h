#ifndef __HIDDEN_NODE__H
#define __HIDDEN_NODE__H

#include "node.h"

class HiddenNode : public Node {
 public:
  HiddenNode();
  double getInput() const;
  double getOutput() const;
  void receiveInput(double input);
};

#endif

