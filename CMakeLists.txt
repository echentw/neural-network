cmake_minimum_required (VERSION 3.1.0 FATAL_ERROR)

project (NeuralNetworkTest)

add_subdirectory (src)
add_subdirectory (test)

target_compile_features(components_test PRIVATE cxx_range_for)
target_compile_features(network_test PRIVATE cxx_range_for)

