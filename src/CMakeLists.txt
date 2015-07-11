set (CMAKE_CXX_STANDARD 11)

add_library (NeuralNetwork neural_network.cpp
                           input_node.cpp hidden_node.cpp output_node.cpp
                           input_layer.cpp hidden_layer.cpp output_layer.cpp
                           weight_matrix.cpp
                           helper.cpp data_reader.cpp)

target_include_directories (NeuralNetwork PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

