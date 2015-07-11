set (CMAKE_CXX_STANDARD 11)

add_executable (components_test components_test.cpp)
add_executable (network_test network_test.cpp)
add_executable (reader_test reader_test.cpp)
add_executable (mnist_test mnist_test.cpp)

target_link_libraries (components_test LINK_PUBLIC NeuralNetwork)
target_link_libraries (network_test LINK_PUBLIC NeuralNetwork)
target_link_libraries (reader_test LINK_PUBLIC NeuralNetwork)
target_link_libraries (mnist_test LINK_PUBLIC NeuralNetwork)

