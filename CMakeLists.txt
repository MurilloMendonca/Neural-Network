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

# Add executable for testing
add_executable(testExecutable
    test/main.cpp
)

# Link your library and dependencies
target_link_libraries(testExecutable PRIVATE libNeuralNet libEasyFileStream)

# Set the output directory for the test executable
set_target_properties(testExecutable PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
)

# Copy out.txt to the binary directory during build
configure_file(test/heart_disease.csv ${CMAKE_CURRENT_BINARY_DIR}/bin/heart_disease.csv COPYONLY)
configure_file(test/iris.csv ${CMAKE_CURRENT_BINARY_DIR}/bin/iris.csv COPYONLY)
configure_file(test/wine.csv ${CMAKE_CURRENT_BINARY_DIR}/bin/wine.csv COPYONLY)

