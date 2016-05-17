#include "weight_matrix.h"

WeightMatrix::WeightMatrix(const Layer& in_layer, const Layer& out_layer,
                           double range_low, double range_high) {
  this->n_rows = in_layer.getSize();
  this->n_cols = out_layer.getSize();

  this->weights.resize(this->n_rows, this->n_cols);
  this->bias_weights.resize(this->n_cols);

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<double> distribution(range_low, range_high);
  for (int j = 0; j < this->n_cols; ++j) {
    this->bias_weights(j) = distribution(generator);
    for (int i = 0; i < this->n_rows; ++i) {
      this->weights(i, j) = distribution(generator);
    }
  }

  boost::multi_array<double, 2>::extent_gen extents;
  this->gradients.resize(extents[n_rows][n_cols]);

  boost::multi_array<double, 1>::extent_gen bias_extents;
  this->bias_gradients.resize(bias_extents[n_cols]);
}

void WeightMatrix::set(int row, int col, double value) {
  this->weights(row, col) = value;
}

void WeightMatrix::setBias(int j, double value) {
  this->bias_weights(j) = value;
}

double WeightMatrix::numRows() const {
  return this->n_rows;
}

double WeightMatrix::numCols() const {
  return this->n_cols;
}

double WeightMatrix::get(int row, int col) const {
  return this->weights(row, col);
}

double WeightMatrix::getBias(int j) const {
  return this->bias_weights(j);
}

std::vector<double> WeightMatrix::fire(const Layer& in_layer) const {
  // this method could be optimized by using dot product
  std::vector<double> output(this->n_cols, 0.0);
  for (int j = 0; j < this->n_cols; ++j) {
    output[j] = bias_weights(j);
    for (int i = 0; i < this->n_rows; ++i) {
      output[j] += in_layer.getOutput(i) * weights(i, j);
    }
  }
  return output;
}

void WeightMatrix::computeGradient(const Layer& in_layer,
                                   const Layer& out_layer) {
  for (int j = 0; j < this->n_cols; ++j) {
    this->bias_gradients[j] = out_layer.getPartialDerivative(j);
    for (int i = 0; i < this->n_rows; ++i) {
      this->gradients[i][j] =
          out_layer.getPartialDerivative(j) * in_layer.getOutput(i);
    }
  }
}

void WeightMatrix::update(double step_size) {
  for (int j = 0; j < this->n_cols; ++j) {
    this->bias_weights(j) -= step_size * this->bias_gradients[j];
    for (int i = 0; i < this->n_rows; ++i) {
      this->weights(i, j) -= step_size * this->gradients[i][j];
    }
  }
}

double WeightMatrix::getPartialDerivative(int i, int j) const {
  return this->gradients[i][j];
}

double WeightMatrix::getBiasPartialDerivative(int j) const {
  return this->bias_gradients[j];
}

