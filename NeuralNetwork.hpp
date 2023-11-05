#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <map>
#include <thread>
#include "Layer.hpp"
#include "ThreadPool.hpp"
#include <future>

class NeuralNetwork {
public:
    NeuralNetwork(std::vector<int> topology,std::string activationFunction = "sigmoid", size_t poolSize = std::thread::hardware_concurrency());
    void forward(const std::vector<double>& inputValues);
    void backpropagate(const std::vector<double>& targetValues);
	std::vector<Layer> getLayers();
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
	ThreadPool pool;
};

