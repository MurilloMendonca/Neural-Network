# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall -O3 -I/usr/include/opencv4 -g
LIBS = -L/usr/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lQt6Widgets -lQt6Test -lQt6OpenGLWidgets -lQt6Gui -lQt6Core
# Targets
all: neural_network

neural_network: main.o NeuralNetwork.o Layer.o Neuron.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

main.o: main.cpp NeuralNetwork.hpp Layer.hpp Neuron.hpp Neuron.cpp Layer.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LIBS)

neural_network.o: NeuralNetwork.cpp NeuralNetwork.hpp Layer.hpp Neuron.hpp Neuron.cpp Layer.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LIBS)

.PHONY: clean
clean:
	rm -f *.o neural_network
