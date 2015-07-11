# neural-network
A C++ neural network library.


Example usage:

// TRAINING

// neural network with input layer size = 784, hidden layer size = 30, output layer size = 10;

NeuralNetwork network(10);

network.addInputLayer(784);

network.addHiddenLayer(30);

network.addOutputLayer();


std::vector<std::vector<double> > input_data; // and fill this with training data

std::vector<std::vector<double> > labels; // and fill this with labels for the training data



network.train(input_data, labels);



// This will save the final parameters after the network is finished training.

// However, the train() method also automatically saves the parameters at certain intervals,

// so you can stop and resume traininng at any point.

network.save("config_file.txt");





// TESTING

std::vector<std::vector<double> > test_input_data; // and fill this with input data for testing

std::vector<std::vector<double> > test_labels; // and fill this with labels for testing



network.load("config_file.txt");

network.test(test_input_data, test_labels);

