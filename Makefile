# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O3 -I/usr/include/opencv4 -g -pthread
LIBS = -L/usr/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lQt6Widgets -lQt6Test -lQt6OpenGLWidgets -lQt6Gui -lQt6Core -lprofiler 
# Targets
all: neural_network

neural_network: main.o NeuralNetwork.o Layer.o Neuron.o ThreadPool.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

main.o: main.cpp NeuralNetwork.hpp Layer.hpp Neuron.hpp Neuron.cpp Layer.cpp ThreadPool.cpp ThreadPool.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LIBS)

neural_network.o: NeuralNetwork.cpp NeuralNetwork.hpp Layer.hpp Neuron.hpp Neuron.cpp Layer.cpp ThreadPool.cpp ThreadPool.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LIBS)

.PHONY: clean
clean:
	rm -f *.o neural_network
