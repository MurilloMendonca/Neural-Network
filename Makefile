# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall -O3 -g

# Targets
all: neural_network

neural_network: main.o NeuralNetwork.o Layer.o Neuron.o
	$(CXX) $(CXXFLAGS) -o $@ $^

main.o: main.cpp NeuralNetwork.hpp Layer.hpp Neuron.hpp Neuron.cpp Layer.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

neural_network.o: NeuralNetwork.cpp NeuralNetwork.hpp Layer.hpp Neuron.hpp Neuron.cpp Layer.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f *.o neural_network
