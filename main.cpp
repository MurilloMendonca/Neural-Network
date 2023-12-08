#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include "NeuralNetwork.hpp"
#include "cpp-easy-file-stream/fs.hpp"


#define _SHOW_LAYERS_ 0
#define _VERBOSE_ 0
#define _TEST_IRIS_ 0
#define _TEST_WINE_ 1
#define _TEST_HEART_ 0
#define __SHOW_FILE_ 0
#define __SHOW_DATASET_ 0
#define __SHOW_OUTPUT_PLOT_ 0

#define DBL_MAX 1.79769e+308
#define DBL_MIN 2.22507e-308
#if __SHOW_OUTPUT_PLOT_
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

void showPlot(NeuralNetwork& nn, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs) {
    int input_dim = inputs[0].size();

    // Define a list of colors for each class
    std::vector<cv::Scalar> colors = { cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(255, 255, 0) };

    // Find min and max for normalization
    double min_val = DBL_MAX;
    double max_val = DBL_MIN;
    for (const auto& input : inputs) {
        for (double val : input) {
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
    }

    for (int i = 0; i < input_dim; ++i) {
        for (int j = i + 1; j < input_dim; ++j) {
            cv::Mat plot_image(500, 500, CV_8UC3, cv::Scalar(200, 200, 200)); 

            // Draw Axes
            cv::line(plot_image, cv::Point(0, 250), cv::Point(500, 250), cv::Scalar(0, 0, 0), 1);  // x-axis
            cv::line(plot_image, cv::Point(250, 0), cv::Point(250, 500), cv::Scalar(0, 0, 0), 1);  // y-axis

            for (size_t idx = 0; idx < inputs.size(); ++idx) {
                nn.forward(inputs[idx]);
                const auto& resultLayer = nn.getLayers().back().neurons;

                int predicted_class = std::distance(resultLayer.begin(), std::max_element(resultLayer.begin(), resultLayer.end(), 
                                    [](const Neuron& a, const Neuron& b) { return a.value < b.value; }));
                int actual_class = std::distance(outputs[idx].begin(), std::max_element(outputs[idx].begin(), outputs[idx].end()));

                // Normalize the inputs to fit within the visualization window
                cv::Point pt((inputs[idx][i] - min_val) / (max_val - min_val) * 500, 
                             (inputs[idx][j] - min_val) / (max_val - min_val) * 500);

                cv::circle(plot_image, pt, 4, colors[actual_class], -1);        // Larger circle for actual data
                cv::circle(plot_image, pt, 2, colors[predicted_class], -1);    // Smaller circle inside for predicted data
            }
            
            // Add labels to axes
            cv::putText(plot_image, "Variable " + std::to_string(j + 1), cv::Point(260, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            cv::putText(plot_image, "Variable " + std::to_string(i + 1), cv::Point(10, 240), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

            // Show the image
            cv::imshow("Input variable " + std::to_string(i + 1) + " vs " + std::to_string(j + 1), plot_image);
        }
    }

    cv::waitKey(0);  // Wait until user presses a key
}


#endif


void showFile(std::string fileName){
	FileStream fs(fileName);
	std::string word;
	while((word = fs.getDelimiter(','))!=""){
		std::cout<<word<<std::endl;
	}
}

void readDataset(std::string fileName,std::vector<std::vector<double>>& inputs,std::vector<std::vector<double>>& outputs, int numberOfCollums, int outputCollum){
	FileStream fs(fileName);
	std::string word;
	std::map<std::string,double> classes;
	word = fs.getDelimiter(',');
	while(word!=""){
		std::vector<double> input;
		for(int i = 0;i<numberOfCollums;i++){
			if(i==outputCollum){
				if(classes.find(word)==classes.end()){
					classes[word] = classes.size();
				}
				outputs.push_back({classes[word]});
				word = fs.getDelimiter(',');

				continue;
			}
			input.push_back(word=="?"?0.0:std::stod(word));
			word = fs.getDelimiter(',');
		}
		inputs.push_back(input);
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

void normalizeInputs(std::vector<std::vector<double>>& inputs){
	std::vector<double> maxValues(inputs[0].size(),DBL_MIN);
	std::vector<double> minValues(inputs[0].size(),DBL_MAX);
	for(auto& x : inputs){
		for(int i = 0;i<x.size();i++){
			maxValues[i] = std::max(maxValues[i],x[i]);
			minValues[i] = std::min(minValues[i],x[i]);
		}
	}
	for(auto& x : inputs){
		for(int i = 0;i<x.size();i++){
			x[i] = (x[i]-minValues[i])/(maxValues[i]-minValues[i]);
		}
	}

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
    int errorNumber = 0;
    std::vector<int> truePositives(outputs[0].size(), 0);  // Initialize true positives for each class to 0
    std::vector<int> falsePositives(outputs[0].size(), 0); // Initialize false positives for each class to 0

    for (size_t i = 0; i < inputs.size(); ++i) {
        nn.forward(inputs[i]);
        const auto& resultLayer = nn.getLayers().back().neurons;

        double exampleError = 0.0;
        int classification = 0;
        for(size_t j = 0; j < outputs[0].size(); j++) {
            double error = outputs[i][j] - resultLayer[j].value;
            exampleError += error * error;  // squared error
            if(resultLayer[j].value > resultLayer[classification].value) {
                classification = j;
            }
        }
        int expectedClassification = 0;
        for(size_t j = 0; j < outputs[0].size(); j++) {
            if(outputs[i][j] > outputs[i][expectedClassification]) {
                expectedClassification = j;
            }
        }
        if(expectedClassification == classification) {
            truePositives[classification]++;
        } else {
            falsePositives[classification]++;
            errorNumber++;
        }
        #if _VERBOSE_
            std::cout << "Input: ";
            for (auto val : inputs[i]) std::cout << val << " ";
            std::cout << "| Expected output: " << expectedClassification << " | Network output: " << classification << std::endl;
        #endif
        totalError += exampleError / outputs[0].size();  // average error for this example
    }

    double mse = totalError / inputs.size();  // mean squared error over all examples
    std::cout << "Mean Squared Error (MSE) on All Data: " << mse << std::endl;

	std::cout<<"Wrong Predictions: "<<errorNumber<<std::endl;
	std::cout<<"Right Predictions: "<<inputs.size()-errorNumber<<std::endl;
	std::cout<<"Accuracy: "<<(double)(inputs.size()-errorNumber)/inputs.size()*100<<"%"<<std::endl;

	// Compute and display precision for each class
	for (size_t i = 0; i < outputs[0].size(); i++) {
		double precision = static_cast<double>(truePositives[i]) / (truePositives[i] + falsePositives[i]);
		std::cout << "Precision for class " << i << ": " << precision * 100 << "%" << std::endl;
	}

}


void trainAndTest(NeuralNetwork& nn, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs, int epochs = 1000) {
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
		readDataset("iris.csv",irisInputs,irisOutputs, 5, 4);
		NeuralNetwork irisNN({4,10,3},"tanh");

		#if _SHOW_LAYERS_
			showTopology(irisNN);
		#endif

		normalizeInputs(irisInputs);
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

		#if __SHOW_OUTPUT_PLOT_
			showPlot(irisNN,irisInputs,irisOutputs);
		#endif

		#if _SHOW_LAYERS_
			showTopology(irisNN);
		#endif
	#endif


	#if _TEST_WINE_
		std::vector<std::vector<double>> wineInputs, wineOutputs;
		readDataset("wine.csv",wineInputs,wineOutputs, 14, 0);
		int numberOfInputs = wineInputs[0].size();
		int numberOfOutputs = wineOutputs[0].size();
		NeuralNetwork wineNN({numberOfInputs,10,numberOfOutputs},"tanh");

		#if _SHOW_LAYERS_
			showTopology(wineNN);
		#endif


		normalizeInputs(wineInputs);
		trainAndTest(wineNN,wineInputs,wineOutputs, 1000);

		//Show Inputs and outputs
		#if __SHOW_DATASET_
			std::cout<<"Inputs: "<<std::endl;
			for(auto& x : wineInputs){
				for(auto& y : x){
					std::cout<<y<<" ";
				}
				std::cout<<std::endl;
			}
			std::cout<<"Outputs: "<<std::endl;
			for(auto& x : wineOutputs){
				for(auto& y : x){
					std::cout<<y<<" ";
				}
				std::cout<<std::endl;
			}
		#endif

		std::cout<<"WINE"<<std::endl;
		run(wineNN,wineInputs,wineOutputs);

		#if __SHOW_OUTPUT_PLOT_
			showPlot(wineNN,wineInputs,wineOutputs);
		#endif

		#if _SHOW_LAYERS_
			showTopology(wineNN);
		#endif
	#endif

	#if _TEST_HEART_
		std::vector<std::vector<double>> heartInputs, heartOutputs;
		readDataset("heart_disease.csv",heartInputs,heartOutputs, 14, 13);
		int numberOfInputs = heartInputs[0].size();
		int numberOfOutputs = heartOutputs[0].size();
		NeuralNetwork heartNN({numberOfInputs,100,numberOfOutputs},"tanh");

		#if _SHOW_LAYERS_
			showTopology(wineNN);
		#endif


		normalizeInputs(heartInputs);
		trainAndTest(heartNN,heartInputs,heartOutputs,100);

		//Show Inputs and outputs
		#if __SHOW_DATASET_
			std::cout<<"Inputs: "<<std::endl;
			for(auto& x : heartInputs){
				for(auto& y : x){
					std::cout<<y<<" ";
				}
				std::cout<<std::endl;
			}
			std::cout<<"Outputs: "<<std::endl;
			for(auto& x : heartOutputs){
				for(auto& y : x){
					std::cout<<y<<" ";
				}
				std::cout<<std::endl;
			}
		#endif

		std::cout<<"WINE"<<std::endl;
		run(heartNN,heartInputs,heartOutputs);

		#if __SHOW_OUTPUT_PLOT_
			showPlot(heartNN,heartInputs,heartOutputs);
		#endif

		#if _SHOW_LAYERS_
			showTopology(heartNN);
		#endif
	#endif



    return 0;
}