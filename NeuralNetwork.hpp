#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <map>
#include "Layer.hpp"

class NeuralNetwork {
public:
    NeuralNetwork(std::vector<int> topology,std::string activationFunction = "sigmoid");
    NeuralNetwork(){};
    void forward(const std::vector<double>& inputValues);
    void backpropagate(const std::vector<double>& targetValues);
	std::vector<Layer> getLayers() const;
	void setLearningRate(double learningRate);

	static std::map<std::string, double(*)(double)> activationFunctions;
	std::string activationFunction;

	static double sigmoid(double x);
    static double sigmoidPrime(double x); 
	static double relu(double x);
    static double reluPrime(double x);
	static double tanh(double x);
	static double tanhPrime(double x);
	
	double activation(double x);
	double activationPrime(double x);

private:
    std::vector<Layer> layers;
    double learningRate = 0.001;
};

