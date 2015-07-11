#ifndef __WEIGHT_MATRIX__H
#define __WEIGHT_MATRIX__H

#include <random>
#include <chrono>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/multi_array.hpp>

#include "layer.h"

class WeightMatrix {
 private:
  // the # of rows of the weight matrix (size of in_layer)
  int n_rows;

  // the # of cols of the weight matrix (size of out_layer)
  int n_cols;

  // weights[i][j] = weight for in_layer[i] and out_layer[j]
  boost::numeric::ublas::matrix<double> weights;

  // bias_weights[j] = bias weight for out_layer[j]
  boost::numeric::ublas::vector<double> bias_weights;

  // gradients[k][i][j] = partial derivative of output k
  //                      w.r.t. weights[i][j]
  boost::multi_array<double, 2> gradients;

  // bias_gradients[k][j] = partial derivative of output k
  //                        w.r.t. bias_weights[j]
  boost::multi_array<double, 1> bias_gradients;

 public:
  // randomizes the matrix using  a uniform distribution
  // in the range [range_low, range_high]
  WeightMatrix(const Layer& in_layer, const Layer& out_layer,
               double range_low=-1.0, double range_high=1.0);

  // helper methods only; not to be used in actual implementation
  void set(int row, int col, double value);
  void setBias(int j, double value);

  // get methods
  double numRows() const;
  double numCols() const;
  double get(int row, int col) const;
  double getBias(int j) const;

  // uses the value of in_layer and computes the output
  std::vector<double> fire(const Layer& in_layer) const;

  // computes the gradients w.r.t. the outputs
  void computeGradient(const Layer& in_layer, const Layer& out_layer);

  // updates the weights given the step size
  void update(double step_size);

  // get method
  // returns the partial derivative of cost w.r.t. weights[i][j]
  double getPartialDerivative(int i, int j) const;

  // get method
  // returns the partial derivative of cost w.r.t. bias_weights[i]
  double getBiasPartialDerivative(int j) const;
};

#endif

