# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall -O3 -g
LIBS = -L/usr/lib 
# Targets
all: neural_network

neural_network: main.o NeuralNetwork.o Layer.o Neuron.o ClassificationNN.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

main.o: main.cpp NeuralNetwork.hpp Layer.hpp Neuron.hpp Neuron.cpp Layer.cpp ClassificationNN.hpp ClassificationNN.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LIBS)

neural_network.o: NeuralNetwork.cpp NeuralNetwork.hpp Layer.hpp Neuron.hpp Neuron.cpp Layer.cpp ClassificationNN.hpp ClassificationNN.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LIBS)

.PHONY: clean
clean:
	rm -f *.o neural_network
