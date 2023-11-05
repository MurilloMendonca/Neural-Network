#include"NeuralNetwork.hpp"

std::map<std::string, double(*)(double)> NeuralNetwork::activationFunctions = {
    {"sigmoid", NeuralNetwork::sigmoid},
    {"relu", NeuralNetwork::relu},
	{"tanh", NeuralNetwork::tanh}
};

NeuralNetwork::NeuralNetwork(std::vector<int> topology, std::string activationFunc, size_t poolSize) 
    : activationFunction(activationFunc), pool(poolSize) {
    for (size_t i = 0; i < topology.size() - 1; ++i) {
        layers.push_back(Layer(topology[i], topology[i + 1], i == 0 ? 0 : topology[i - 1]));
    }
    layers.push_back(Layer(topology.back(), 0, topology[topology.size() - 2]));
}

void NeuralNetwork::forward(const std::vector<double>& inputValues) {
    // Assign the input values to the input layer
    for (size_t i = 0; i < inputValues.size(); ++i) {
        layers[0].neurons[i].value = inputValues[i];
    }



    for (size_t layerNum = 1; layerNum < layers.size(); ++layerNum) {
        Layer& prevLayer = layers[layerNum - 1];
        Layer& currentLayer = layers[layerNum];
        for (Neuron& neuron : currentLayer.neurons) {

                    double sum = 0.0;
                    for (size_t weightIndex = 0; weightIndex < neuron.weights.size(); ++weightIndex) {
                        sum += prevLayer.neurons[weightIndex].value * neuron.weights[weightIndex];
                    }
                    neuron.value = this->activation(sum + neuron.bias);

        }
    }

}



void NeuralNetwork::backpropagate(const std::vector<double>& targetValues) {
    // 1. Compute the output layer's error
    Layer& outputLayer = layers.back();
    double errorSumSquared = 0.0;

    for (size_t n = 0; n < outputLayer.neurons.size(); ++n) {
        double delta = targetValues[n] - outputLayer.neurons[n].value;
        errorSumSquared += delta * delta;
        outputLayer.neurons[n].delta = delta * activationPrime(outputLayer.neurons[n].value);
    }

    std::vector<std::future<void>> futures;

    // Calculate the number of available threads in the pool
    size_t poolSize = pool.threadCount();

    // 2. Propagate the error backwards through the network
    for (size_t layerNum = layers.size() - 2; layerNum > 0; --layerNum) {
        Layer& currentLayer = layers[layerNum];
        Layer& nextLayer = layers[layerNum + 1];

        size_t chunkSize = currentLayer.neurons.size() / poolSize;
        size_t remainingNeurons = currentLayer.neurons.size() % poolSize;

        for (size_t t = 0; t < poolSize; ++t) {
            size_t startIdx = t * chunkSize;
            size_t endIdx = (t + 1) * chunkSize + (t == poolSize - 1 ? remainingNeurons : 0);

            futures.emplace_back(
                pool.enqueue([startIdx, endIdx, layerNum, this, &currentLayer, &nextLayer]() {
                    for (size_t n = startIdx; n < endIdx; ++n) {
                        double deltaSum = 0.0;
                        Neuron& neuron = currentLayer.neurons[n];
                        for (Neuron& nextNeuron : nextLayer.neurons) {
                            deltaSum += nextNeuron.delta * neuron.weights[layerNum];
                        }
                        neuron.delta = deltaSum * activationPrime(neuron.value);
                    }
                })
            );
        }
    }

    // Wait for all tasks to finish
    for (auto& future : futures) {
        future.get();
    }

    futures.clear();

    // 3. Update weights and biases using the calculated errors (Parallelized)
    for (size_t layerNum = layers.size() - 1; layerNum > 0; --layerNum) {
        Layer& currentLayer = layers[layerNum];
        Layer& prevLayer = layers[layerNum - 1];

        size_t chunkSize = currentLayer.neurons.size() / poolSize;
        size_t remainingNeurons = currentLayer.neurons.size() % poolSize;

        for (size_t t = 0; t < poolSize; ++t) {
            size_t startIdx = t * chunkSize;
            size_t endIdx = (t + 1) * chunkSize + (t == poolSize - 1 ? remainingNeurons : 0);

            futures.emplace_back(
                pool.enqueue([startIdx, endIdx, layerNum, this, &currentLayer, &prevLayer]() {
                    for (size_t n = startIdx; n < endIdx; ++n) {
                        Neuron& neuron = currentLayer.neurons[n];
                        for (size_t w = 0; w < neuron.weights.size(); ++w) {
                            neuron.weights[w] += this->learningRate * neuron.delta * prevLayer.neurons[w].value;
                        }
                        neuron.bias += this->learningRate * neuron.delta;
                    }
                })
            );
        }
    }

    // Wait for all tasks to finish
    for (auto& future : futures) {
        future.get();
    }
}


double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double NeuralNetwork::sigmoidPrime(double x) {
    double y = sigmoid(x);
    return y * (1.0 - y);
}

double NeuralNetwork::relu(double x) {
    return x > 0.0 ? x : 0.0;
}

double NeuralNetwork::reluPrime(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

double NeuralNetwork::tanh(double x) {
	return std::tanh(x);
}

double NeuralNetwork::tanhPrime(double x) {
	return 1.0 - std::tanh(x) * std::tanh(x);
}

double NeuralNetwork::activation(double x) {
    return activationFunctions[activationFunction](x);
}

double NeuralNetwork::activationPrime(double x) {
    if (activationFunction == "sigmoid") return sigmoidPrime(x);
    if (activationFunction == "relu") return reluPrime(x);
    if (activationFunction == "tanh") return tanhPrime(x);
    throw std::runtime_error("Unknown activation function");
}

std::vector<Layer> NeuralNetwork::getLayers() {
    return layers;
}

void NeuralNetwork::setLearningRate(double lr) {
    learningRate = lr;
}
