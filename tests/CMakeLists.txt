set (CMAKE_CXX_STANDARD 11)

include_directories(${NeuralNetwork_SOURCE_DIR}/nn/include)

add_executable (components_test components_test.cpp)
add_executable (network_test network_test.cpp)
add_executable (reader_test reader_test.cpp)
add_executable (mnist_test mnist_test.cpp)

target_link_libraries (components_test LINK_PUBLIC neural_network)
target_link_libraries (network_test LINK_PUBLIC neural_network)
target_link_libraries (reader_test LINK_PUBLIC neural_network)
target_link_libraries (mnist_test LINK_PUBLIC neural_network)
