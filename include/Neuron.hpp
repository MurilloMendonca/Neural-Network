#pragma once
#include <vector>
#include <cmath>
#include <random>

class Neuron {
public:
    Neuron(int numInputs,int numOutputs);

	static std::mt19937 generator;
    static std::uniform_real_distribution<double> distribution;
	std::vector<double> weights;

    double value;
    double delta;
	double bias;
};


