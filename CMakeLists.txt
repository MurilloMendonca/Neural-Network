cmake_minimum_required(VERSION 3.12)
project(neuralNet VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add your own library
add_library(libNeuralNet
    src/ClassificationNN.cpp
    src/Layer.cpp
    src/NeuralNetwork.cpp
    src/Neuron.cpp
)

target_include_directories(libNeuralNet PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add dependencies
add_subdirectory(dependencies/cpp-easy-file-stream)

