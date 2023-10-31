#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include "NeuralNetwork.hpp"
#include "cpp-easy-file-stream/fs.hpp"


#define _SHOW_LAYERS_ 0
#define _VERBOSE_ 1
#define _TEST_IRIS_ 1
#define __SHOW_FILE_ 0
#define __SHOW_DATASET_ 0

void showFile(std::string fileName){
	FileStream fs(fileName);
	std::string word;
	while((word = fs.getDelimiter(','))!=""){
		std::cout<<word<<std::endl;
	}
}

void readDataset(std::string fileName,std::vector<std::vector<double>>& inputs,std::vector<std::vector<double>>& outputs){
	FileStream fs(fileName);
	std::string word;
	std::map<std::string,double> classes;
	while((word = fs.getDelimiter(','))!=""){
		inputs.push_back({stof(word),stof(fs.getDelimiter(',')),stof(fs.getDelimiter(',')),stof(fs.getDelimiter(','))});
		std::string className = fs.getDelimiter(',');
		if(classes.find(className)==classes.end()){
			classes[className] = classes.size();
		}
		outputs.push_back({classes[className]});
	}

	for(auto& x : outputs){
		int classification = x[0];
		x.clear();
		for(int i = 0;i<classes.size();i++){
			if(i==classification){
				x.push_back(1);
			}else{
				x.push_back(0);
			}
		}
	}
	#if _VERBOSE_
	std::cout<<"Classes: "<<std::endl;
	for(auto it = classes.begin();it!=classes.end();it++){
		std::cout<<it->first<<" "<<it->second<<std::endl;
	}
	#endif
}

void train(NeuralNetwork& nn, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs, int epochs = 10000) {
    for (int i = 0; i < epochs; ++i) {
        for (size_t j = 0; j < inputs.size(); ++j) {
            nn.forward(inputs[j]);
            nn.backpropagate(outputs[j]);
        }
    }
}

void run(NeuralNetwork& nn, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs) {
    double totalError = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        nn.forward(inputs[i]);
        const auto& resultLayer = nn.getLayers().back().neurons;

        double exampleError = 0.0;
		int classification = 0;
        for(size_t j = 0; j < outputs[0].size(); j++) {
            double error = outputs[i][j] - resultLayer[j].value;
            exampleError += error * error;  // squared error
			if(resultLayer[j].value>resultLayer[classification].value){
				classification = j;
			}
			
        }
		#if _VERBOSE_
			int expectedClassification = 0;
			for(size_t j = 0; j < outputs[0].size(); j++) {
				if(outputs[i][j]>outputs[i][expectedClassification]){
					expectedClassification = j;
				}
			}
            std::cout << "Input: ";
            for (auto val : inputs[i]) std::cout << val << " ";
            std::cout << "| Expected output: " << expectedClassification << " | Network output: " << classification << std::endl;
		#endif
        totalError += exampleError / outputs[0].size();  // average error for this example
    }
    double mse = totalError / inputs.size();  // mean squared error over all examples
    std::cout << "Mean Squared Error (MSE) on All Data: " << mse << std::endl;
}

void trainAndTest(NeuralNetwork& nn, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs, int epochs = 10000) {
    // Shuffle the dataset
    auto shuffled_indices = std::vector<size_t>(inputs.size());
    std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0);  // Fill with 0, 1, ..., n-1
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), std::default_random_engine(seed));

    // Split data into 80% training and 20% test
    size_t trainingSize = inputs.size() * 0.8;
    std::vector<std::vector<double>> trainingInputs(trainingSize);
    std::vector<std::vector<double>> trainingOutputs(trainingSize);
    std::vector<std::vector<double>> testInputs(inputs.size() - trainingSize);
    std::vector<std::vector<double>> testOutputs(outputs.size() - trainingSize);

    for (size_t i = 0; i < trainingSize; ++i) {
        trainingInputs[i] = inputs[shuffled_indices[i]];
        trainingOutputs[i] = outputs[shuffled_indices[i]];
    }
    for (size_t i = trainingSize; i < inputs.size(); ++i) {
        testInputs[i - trainingSize] = inputs[shuffled_indices[i]];
        testOutputs[i - trainingSize] = outputs[shuffled_indices[i]];
    }

    // Training and testing
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Train on training data
        for (size_t i = 0; i < trainingInputs.size(); ++i) {
            nn.forward(trainingInputs[i]);
            nn.backpropagate(trainingOutputs[i]);  // Assuming your NN has a backprop method
        }

        // Test on test data and compute MSE
		#if _VERBOSE_
			double totalError = 0.0;
			for (size_t i = 0; i < testInputs.size(); ++i) {
				nn.forward(testInputs[i]);
				const auto& resultLayer = nn.getLayers().back().neurons;
				
				double exampleError = 0.0;
				for (size_t j = 0; j < testOutputs[0].size(); j++) {
					double error = testOutputs[i][j] - resultLayer[j].value;
					exampleError += error * error;  // squared error
				}
				totalError += exampleError / testOutputs[0].size();  // average error for this example
			}
			double mse = totalError / testInputs.size();
			std::cout << "Epoch " << (epoch + 1) << " - Mean Squared Error (MSE) on Test Data: " << mse << std::endl;
		#endif
    }
}

void showTopology(NeuralNetwork& nn) {
    std::cout << "Network topology:" << std::endl;
    int layerNum = 1; // For display purposes
    for (const auto& layer : nn.getLayers()) {
        std::cout << "Layer " << layerNum << " (" << layer.neurons.size() << " neurons):" << std::endl;
        int neuronNum = 1; // For display purposes
        for (const auto& neuron : layer.neurons) {
            std::cout << "\tNeuron " << neuronNum << " weights: ";
            for (const auto& weight : neuron.weights) {
                std::cout << weight << " ";
            }
            std::cout << "| Bias: " << neuron.bias << std::endl;  // Also displaying the bias for completeness
            neuronNum++;
        }
        layerNum++;
    }
    std::cout << std::endl;
}

int main() {
	#if __SHOW_FILE_
	showFile("iris.csv");
	#endif

	#if _TEST_IRIS_
		std::vector<std::vector<double>> irisInputs, irisOutputs;
		readDataset("iris.csv",irisInputs,irisOutputs);
		NeuralNetwork irisNN({4,8,8,3},"tanh");

		#if _SHOW_LAYERS_
			showTopology(irisNN);
		#endif


		trainAndTest(irisNN,irisInputs,irisOutputs);

		//Show Inputs and outputs
		#if __SHOW_DATASET_
			std::cout<<"Inputs: "<<std::endl;
			for(auto& x : irisInputs){
				for(auto& y : x){
					std::cout<<y<<" ";
				}
				std::cout<<std::endl;
			}
			std::cout<<"Outputs: "<<std::endl;
			for(auto& x : irisOutputs){
				for(auto& y : x){
					std::cout<<y<<" ";
				}
				std::cout<<std::endl;
			}
		#endif

		std::cout<<"IRIS"<<std::endl;
		run(irisNN,irisInputs,irisOutputs);

		#if _SHOW_LAYERS_
			showTopology(irisNN);
		#endif
	#endif



    return 0;
}