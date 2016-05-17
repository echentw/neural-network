set(CMAKE_CXX_STANDARD 11)

include_directories(${NeuralNetwork_SOURCE_DIR}/nn/include)

add_library(neural_network neural_network.cpp
                           input_node.cpp hidden_node.cpp output_node.cpp
                           input_layer.cpp hidden_layer.cpp output_layer.cpp
                           weight_matrix.cpp
                           helper.cpp data_reader.cpp)
