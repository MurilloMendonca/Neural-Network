#pragma once
#include <vector>
#include <cmath>
#include <random>
#include "Neuron.hpp"
class Layer {
public:
	Layer(int size, int numOutputs, int numInputs);

    std::vector<Neuron> neurons;
};
