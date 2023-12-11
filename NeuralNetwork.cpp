#include"NeuralNetwork.hpp"
#include <hip/hip_runtime.h>
#include "Kernels.hpp"
std::map<std::string, double(*)(double)> NeuralNetwork::activationFunctions = {
    {"sigmoid", NeuralNetwork::sigmoid},
    {"relu", NeuralNetwork::relu},
	{"tanh", NeuralNetwork::tanh}
};

NeuralNetwork::NeuralNetwork(std::vector<int> topology, std::string activationFunc) 
    : activationFunction(activationFunc) {
    for (size_t i = 0; i < topology.size() - 1; ++i) {
        layers.push_back(Layer(topology[i], topology[i + 1], i == 0 ? 0 : topology[i - 1]));
    }
    layers.push_back(Layer(topology.back(), 0, topology[topology.size() - 2]));
}

void NeuralNetwork::forward(const std::vector<double>& inputValues) {
    for (size_t i = 0; i < inputValues.size(); ++i) {
        layers[0].neurons[i].value = inputValues[i];
    }
    prepareDataForHip();

    // Forward propagation
    //for (size_t layerNum = 1; layerNum < layers.size(); ++layerNum) {
    //    Layer& prevLayer = layers[layerNum - 1];
    //    for (Neuron& neuron : layers[layerNum].neurons) {
    //        double sum = 0.0;
    //        size_t weightIndex = 0; // index for accessing weights of the neuron
    //        for (Neuron& prevNeuron : prevLayer.neurons) {
    //            sum += prevNeuron.value * neuron.weights[weightIndex]; // Access weights from the current neuron, not the previous one
    //            weightIndex++; // Move to the next weight for the next neuron in the previous layer
    //        }
    //        neuron.value = activation(sum + neuron.bias);
    //    }
    //}

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

    // 2. Propagate the error backwards through the network
    for (size_t layerNum = layers.size() - 2; layerNum > 0; --layerNum) {
        Layer& currentLayer = layers[layerNum];
        Layer& nextLayer = layers[layerNum + 1];

        for (Neuron& neuron : currentLayer.neurons) {
            double deltaSum = 0.0;
            for (Neuron& nextNeuron : nextLayer.neurons) {
				if(layerNum<neuron.weights.size())
                	deltaSum += nextNeuron.delta * neuron.weights[layerNum];
            }
            neuron.delta = deltaSum * activationPrime(neuron.value);
        }
    }

    // 3. Update weights and biases using the calculated errors
    for (size_t layerNum = layers.size() - 1; layerNum > 0; --layerNum) {
        Layer& currentLayer = layers[layerNum];
        Layer& prevLayer = layers[layerNum - 1];

        for (Neuron& neuron : currentLayer.neurons) {
            for (size_t w = 0; w < neuron.weights.size(); ++w) {
                neuron.weights[w] += learningRate * neuron.delta * prevLayer.neurons[w].value;
            }
            neuron.bias += learningRate * neuron.delta; // Update the bias
        }
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

std::vector<Layer> NeuralNetwork::getLayers() const{
    return layers;
}

void NeuralNetwork::setLearningRate(double lr) {
    learningRate = lr;
}

void NeuralNetwork::prepareDataForHip() {
    size_t totalNeurons = 0;
    size_t totalWeights = 0;

    // Calculate the total number of neurons and weights
    for (const auto& layer : layers) {
        totalNeurons += layer.neurons.size();
        for (const auto& neuron : layer.neurons) {
            totalWeights += neuron.weights.size();
        }
    }

    // Allocate host memory for neuron values, weights, biases
    double* neuronValues = new double[totalNeurons];
    double* weights = new double[totalWeights];
    double* biases = new double[totalNeurons];

    // Flatten neuron data
    size_t neuronIndex = 0;
    size_t weightIndex = 0;
    for (const auto& layer : layers) {
        for (const auto& neuron : layer.neurons) {
            neuronValues[neuronIndex] = neuron.value;
            biases[neuronIndex] = neuron.bias;
            for (double weight : neuron.weights) {
                weights[weightIndex++] = weight;
            }
            neuronIndex++;
        }
    }

    // Allocate device memory and copy data
    double *d_neuronValues, *d_weights, *d_biases;
    hipMalloc(&d_neuronValues, totalNeurons * sizeof(double));
    hipMalloc(&d_weights, totalWeights * sizeof(double));
    hipMalloc(&d_biases, totalNeurons * sizeof(double));

    hipMemcpy(d_neuronValues, neuronValues, totalNeurons * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_weights, weights, totalWeights * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_biases, biases, totalNeurons * sizeof(double), hipMemcpyHostToDevice);

    // Additional arrays for layer inputs and outputs
    double *d_layerInput, *d_layerOutput;
    size_t inputSize = layers[0].neurons.size(); // Size of input layer
    size_t outputSize; // Will be set for each layer

    size_t weightOffsetElements = 0; // Offset in terms of number of elements
    size_t biasOffsetElements = 0;

    // Allocate host memory for all layers' outputs
    double* allLayersOutput = new double[totalNeurons];
    size_t outputOffset = 0; // Offset to track the output array position

    // Allocate memory for layer input (initialized with network input)
    hipMalloc(&d_layerInput, inputSize * sizeof(double));
    hipMemcpy(d_layerInput, neuronValues, inputSize * sizeof(double), hipMemcpyHostToDevice);

    

    // For each layer, perform forward pass
    for (size_t layerNum = 1; layerNum < layers.size(); ++layerNum) {
        outputSize = layers[layerNum].neurons.size();
        

        //std::cout<<"\n\nLayer "<<layerNum<<": "<<inputSize<<" -> "<<outputSize;

        // Allocate memory for layer output
        hipMalloc(&d_layerOutput, outputSize * sizeof(double));

        // Kernel configuration
        int threadsPerBlock = 512;
        int blocksPerGrid = (outputSize + threadsPerBlock - 1) / threadsPerBlock;

        // Call the kernel
        forwardPassKernel<<<blocksPerGrid, threadsPerBlock>>>(d_layerInput, d_weights, d_biases, d_layerOutput, inputSize, outputSize, weightOffsetElements, biasOffsetElements);

        hipMemcpy(allLayersOutput + outputOffset, d_layerOutput, outputSize * sizeof(double), hipMemcpyDeviceToHost);

        //std::cout<<"\nOutput: ";
        //for (size_t i = 0; i < outputSize; i++) {
        //    std::cout<<allLayersOutput[i+outputOffset]<<" ";
        //}
        //std::cout<<"\n";

        outputOffset += outputSize;

        // Calculate the next offsets
        weightOffsetElements += layers[layerNum - 1].neurons.size() * layers[layerNum].neurons.size();
        biasOffsetElements += layers[layerNum].neurons.size();

        // Free the input of the previous layer and use the output as the next layer's input
        if (layerNum > 1) {
            hipFree(d_layerInput);
        }
        d_layerInput = d_layerOutput;
        inputSize = outputSize;
    }

    // Update the neuron values in each layer
    neuronIndex = 0;
    for (size_t layerNum = 1; layerNum < layers.size(); ++layerNum){
        Layer& layer = layers[layerNum];
        for (auto& neuron : layer.neurons) {
            neuron.value = allLayersOutput[neuronIndex++];
        }
    }

    // Show all layers' outputs
    //std::cout << "\n\nAll layers' outputs:" << std::endl;
    //for (size_t i = 0; i < totalNeurons; ++i) {
    //    std::cout << allLayersOutput[i] << " ";
    //}

    // Free host memory for the all layers' outputs
    delete[] allLayersOutput;

    // Free host memory
    delete[] neuronValues;
    delete[] weights;
    delete[] biases;

    // Free device memory
    hipFree(d_layerInput); // Free the last layer's output (same as d_layerOutput)
    hipFree(d_neuronValues);
    hipFree(d_weights);
    hipFree(d_biases);
}