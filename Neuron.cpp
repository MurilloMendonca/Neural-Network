#include "Neuron.hpp"
// Define and initialize the random number generator and distribution
std::mt19937 Neuron::generator(std::random_device{}()); // You might want to seed it differently
std::uniform_real_distribution<double> Neuron::distribution(-1.0, 1.0);

Neuron::Neuron(int numInputs, int numOutputs)
    : value(0), delta(0) 
{
    // Initialize the weights for the number of inputs
    for (int i = 0; i < numInputs; ++i) {
        weights.push_back(distribution(generator));
    }
    // Initialize the bias
    bias = distribution(generator);
}
