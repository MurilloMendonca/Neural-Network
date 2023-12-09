#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>

#include "cpp-easy-file-stream/fs.hpp"
#include "NeuralNetwork.hpp"


using inputsType = std::vector<std::vector<double>>;
using outputsType = std::vector<std::vector<double>>;
using topologyType = std::vector<int>;

class ClassificationNN : public NeuralNetwork {
	private:
		inputsType inputs;
		outputsType outputs;
		void readDataset(const std::string& fileName, int numberOfCollums, int indexOfClassCollumn);
		void normalizeInputs();
		
	public:
		ClassificationNN();
		ClassificationNN(const topologyType& topology, const inputsType& inputs,
						const outputsType& outputs,std::string activationFunction = "sigmoid");
		ClassificationNN(const topologyType& topology, const std::string& fileName, int numberOfCollums,
						int indexOfClassCollumn, std::string activationFunction = "sigmoid");

		void train(int epochs);
		float test();
		void trainAndTest(int epochs);
		void testWithOutput();
		void showTopology();

		inputsType getInputs() const;
		outputsType getOutputs() const;
};