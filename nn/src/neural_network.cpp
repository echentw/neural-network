#include "neural_network.h"

NeuralNetwork::NeuralNetwork(int n_outputs)
  : input_layer(0),
    output_layer(n_outputs) {
  assert(n_outputs > 0);
  this->n_outputs = n_outputs;
}

void NeuralNetwork::addInputLayer(int size) {
  this->input_layer = InputLayer(size);
}

void NeuralNetwork::addHiddenLayer(int size) {
  hidden_layers.push_back(HiddenLayer(size));
  int n_hlayers = hidden_layers.size();
  if (n_hlayers == 1) {
    weight_matrices.push_back(
        WeightMatrix(input_layer, hidden_layers[n_hlayers-1]));
  } else {
    weight_matrices.push_back(
        WeightMatrix(hidden_layers[n_hlayers-2], hidden_layers[n_hlayers-1]));
  }
}

void NeuralNetwork::addOutputLayer() {
  int n_hlayers = hidden_layers.size();
  if (n_hlayers == 0) {
    weight_matrices.push_back(
        WeightMatrix(input_layer, output_layer));
  } else {
    weight_matrices.push_back(
        WeightMatrix(hidden_layers[n_hlayers-1], output_layer));
  }
}

double NeuralNetwork::getIterationNumber() const {
  return this->iteration;
}

double NeuralNetwork::getBatchSize() const {
  return this->batch_size;
}

double NeuralNetwork::getStepSize() const {
  return this->step_size;
}

double NeuralNetwork::getThreshold() const {
  return this->threshold;
}

void NeuralNetwork::setBatchSize(double batch_size) {
  this->batch_size = batch_size;
}

void NeuralNetwork::setStepSize(double step_size) {
  this->step_size = step_size;
}

void NeuralNetwork::setThreshold(double threshold) {
  this->threshold = threshold;
}

std::vector<double> NeuralNetwork::computeOutput(std::vector<double> input) {
  this->input_layer.receiveInput(input);
  for (int i = 0; i < weight_matrices.size(); ++i) {
    std::vector<double> transition;
    if (i == 0) {
      transition = weight_matrices[i].fire(input_layer);
      hidden_layers[i].receiveInput(transition);
    } else if (i < weight_matrices.size() - 1) {
      transition = weight_matrices[i].fire(hidden_layers[i-1]);
      hidden_layers[i].receiveInput(transition);
    } else {
      transition = weight_matrices[i].fire(hidden_layers[i-1]);
      output_layer.receiveInput(transition);
    }
  }
  return output_layer.getOutput();
}

void NeuralNetwork::backpropagate(std::vector<double> correct_output) {
  output_layer.computeGradient(correct_output);
  if (hidden_layers.empty()) {
    weight_matrices[0].computeGradient(input_layer, output_layer);
    weight_matrices[0].update(step_size);
  } else {
    // computing all the gradients
    int index = hidden_layers.size() - 1;
    weight_matrices[index + 1].computeGradient(
        hidden_layers[index], output_layer);
    hidden_layers[index].computeGradient(
        weight_matrices[index + 1], output_layer);
    for (int i = hidden_layers.size() - 2; i >= 0; --i) {
      weight_matrices[i + 1].computeGradient(
          hidden_layers[i], hidden_layers[i + 1]);
      hidden_layers[i].computeGradient(
          weight_matrices[i + 1], hidden_layers[i + 1]);
    }
    weight_matrices[0].computeGradient(input_layer, hidden_layers[0]);

    // updating all the parameters
    for (int i = 0; i < weight_matrices.size(); ++i) {
      weight_matrices[i].update(step_size);
    }
  }
}

void NeuralNetwork::train(std::vector<std::vector<double> > inputs,
                          std::vector<std::vector<double> > outputs,
                          int save_period,
                          std::string save_filename) {
  assert(inputs.size() == outputs.size());

  srand(time(NULL));
  std::ofstream error_file;
  error_file.open("error.csv");
  error_file << "batch_#,L2_error" << std::endl;

  int n_training_data = inputs.size();
  double average_error = 2 * threshold;
  while (average_error > threshold) {
    ++iteration;

    average_error = 0.0;
    for (int i = 0; i < batch_size; ++i) {
      int id = rand() % n_training_data;
      std::vector<double> computed_output = computeOutput(inputs[id]);
      backpropagate(outputs[id]);
      average_error += computeError(computed_output, outputs[id]);
    }

    // write error to excel file
    average_error /= batch_size;
    error_file << iteration << "," << average_error << std::endl;

    std::cout << "batch " << iteration << ": "
              << "error = " << average_error << std::endl;

    if (save_period != -1 && (iteration % save_period == 0)) {
        save(save_filename);
    }
  }
}

void NeuralNetwork::test(std::vector<std::vector<double> > inputs,
                         std::vector<std::vector<double> > labels) {
  assert(inputs.size() == labels.size());

  int num_total = inputs.size();
  int num_correct = 0;
  for (int i = 0; i < inputs.size(); ++i) {
    std::vector<double> output = computeOutput(inputs[i]);
    int id = getIndexWithMaxValue(output);
    if (labels[i][id] > 0) {
      ++num_correct;
    }
  }

  std::cout << "Classified " << num_correct << " out of " << num_total
            << " correctly.\n";
  std::cout << "Accuracy: " << 100.0 * num_correct / num_total << "%\n";
}

void NeuralNetwork::save(std::string filename) const {
  std::ofstream config_file(filename);

  config_file << "Iteration number\n" << iteration << "\n\n";
  config_file << "Batch size\n" << batch_size << "\n\n";
  config_file << "Step size\n" << step_size << "\n\n";
  config_file << "Threshold\n" << threshold << "\n\n";

  config_file << "Input layer size\n"
              << input_layer.getSize() << "\n\n";

  config_file << "Hidden layer sizes\n";
  config_file << hidden_layers.size() << std::endl;
  for (int i = 0; i < hidden_layers.size(); ++i) {
    config_file << hidden_layers[i].getSize() << " ";
  }
  config_file << "\n\n";

  config_file << "Output layer size" << std::endl
              << output_layer.getSize() << "\n\n";

  for (int i = 0; i < weight_matrices.size(); ++i) {
    config_file << "Weight matrix " << i << std::endl;
    config_file << weight_matrices[i].numRows() << " "
                << weight_matrices[i].numCols() << std::endl;
    for (int r = 0; r < weight_matrices[i].numRows(); ++r) {
      for (int c = 0; c < weight_matrices[i].numCols(); ++c) {
        config_file << weight_matrices[i].get(r, c) << " ";
      }
      config_file << std::endl;
    }
    for (int j = 0; j < weight_matrices[i].numCols(); ++j) {
      config_file << weight_matrices[i].getBias(j) << " ";
    }
    config_file << "\n\n";
  }

  config_file.close();
}

void NeuralNetwork::load(std::string filename) {
  std::ifstream config_file(filename);
  std::string line;

  while (std::getline(config_file, line)) {
    if (line == "Iteration number") {
      std::getline(config_file, line);
      iteration = atoi(line.c_str());
      std::getline(config_file, line);

    } else if (line == "Batch size") {
      std::getline(config_file, line);
      batch_size = atoi(line.c_str());
      std::getline(config_file, line);

    } else if (line == "Step size") {
      std::getline(config_file, line);
      step_size = atof(line.c_str());
      std::getline(config_file, line);

    } else if (line == "Threshold") {
      std::getline(config_file, line);
      threshold = atof(line.c_str());
      std::getline(config_file, line);

    } else if (line == "Input layer size") {
      std::getline(config_file, line);
      int input_layer_size = atoi(line.c_str());
      addInputLayer(input_layer_size);
      std::getline(config_file, line);

    } else if (line == "Hidden layer sizes") {
      hidden_layers.clear();
      std::getline(config_file, line);
      int num_hidden_layers = atoi(line.c_str());
      std::getline(config_file, line);
      std::vector<int> layer_sizes = parseInts(line, ' ');

      assert(layer_sizes.size() == num_hidden_layers);

      for (int i = 0; i < layer_sizes.size(); ++i) {
        addHiddenLayer(layer_sizes[i]);
      }
      std::getline(config_file, line);

    } else if (line == "Output layer size") {
      std::getline(config_file, line);
      n_outputs = atoi(line.c_str());
      output_layer.adjust(n_outputs);
      addOutputLayer();
      std::getline(config_file, line);

    } else {
      // weight matrix
      std::string prefix = "Weight matrix ";
      assert(line.substr(0, prefix.size()) == prefix);
      int id = atoi(
          line.substr(prefix.size(), line.size() - prefix.size()).c_str());

      std::getline(config_file, line);
      std::vector<int> dimensions = parseInts(line, ' ');
      assert(dimensions.size() == 2);

      if (hidden_layers.empty()) {
        assert(input_layer.getSize() == dimensions[0]);
        assert(output_layer.getSize() == dimensions[1]);
      } else if (id == 0) {
        assert(input_layer.getSize() == dimensions[0]);
        assert(hidden_layers[id].getSize() == dimensions[1]);
      } else if (id == hidden_layers.size()) {
        assert(hidden_layers[id - 1].getSize() == dimensions[0]);
        assert(output_layer.getSize() == dimensions[1]);
      } else {
        assert(hidden_layers[id - 1].getSize() == dimensions[0]);
        assert(hidden_layers[id].getSize() == dimensions[1]);
      }

      for (int i = 0; i < dimensions[0]; ++i) {
        std::getline(config_file, line);
        std::vector<double> row = parseDoubles(line, ' ');
        assert(row.size() == dimensions[1]);
        for (int j = 0; j < dimensions[1]; ++j) {
          weight_matrices[id].set(i, j, row[j]);
        }
      }
      std::getline(config_file, line);
      std::vector<double> col = parseDoubles(line, ' ');
      for (int j = 0; j < dimensions[1]; ++j) {
        weight_matrices[id].setBias(j, col[j]);
      }

      std::getline(config_file, line);
    }
  }

  config_file.close();
}

