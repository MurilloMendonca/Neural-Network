#include "Layer.hpp"
Layer::Layer(int size, int numOutputs, int numInputs) {
    for (int i = 0; i < size; ++i) {
        neurons.push_back(Neuron(numInputs, numOutputs));
    }
}