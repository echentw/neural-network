#ifndef __OUTPUT_NODE__H
#define __OUTPUT_NODE__H

#include "node.h"

class OutputNode : public Node {
 public:
  OutputNode();
  double getInput() const;
  double getOutput() const;
  void receiveInput(double input);
};

#endif

