#include <iostream>
#include <vector>
#include "../dependencies/cpp-easy-file-stream/include/fs.hpp"
#include "../include/ClassificationNN.hpp"


#define _SHOW_LAYERS_ 0
#define _VERBOSE_ 0
#define _TEST_IRIS_ 1
#define _TEST_WINE_ 0
#define __SHOW_FILE_ 0
#define __SHOW_DATASET_ 0



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


int main() {
	

	#if _TEST_IRIS_
		#if __SHOW_FILE_
		showFile("iris.csv");
		#endif
		ClassificationNN irisNN({4,10,3},"iris.csv",5,4,"tanh");

		#if _SHOW_LAYERS_
			irisNN.showTopology();
		#endif

		
		irisNN.trainAndTest(1000);

		//Show Inputs and outputs
		#if __SHOW_DATASET_
			std::cout<<"Inputs: "<<std::endl;
			for(auto& x : irisNN.getInputs()){
				for(auto& y : x){
					std::cout<<y<<" ";
				}
				std::cout<<std::endl;
			}
			std::cout<<"Outputs: "<<std::endl;
			for(auto& x : irisNN.getOutputs()){
				for(auto& y : x){
					std::cout<<y<<" ";
				}
				std::cout<<std::endl;
			}
		#endif

		std::cout<<"IRIS"<<std::endl;
		irisNN.testWithOutput();

		#if _SHOW_LAYERS_
			irisNN.showTopology();
		#endif
	#endif


	#if _TEST_WINE_
		std::vector<std::vector<double>> wineInputs, wineOutputs;
		readDataset("wine.csv",wineInputs,wineOutputs, 14, 0);
		int numberOfInputs = wineInputs[0].size();
		int numberOfOutputs = wineOutputs[0].size();
		//NeuralNetwork wineNN({numberOfInputs,10,numberOfOutputs},"tanh");

		ClassificationNN wineNN({numberOfInputs,10,numberOfOutputs},"wine.csv",14,0,"tanh");

		#if _SHOW_LAYERS_
			wineNN.showTopology();
		#endif


		//normalizeInputs(wineInputs);
		//trainAndTest(wineNN,wineInputs,wineOutputs, 1000);
		wineNN.trainAndTest(1000);

		//Show Inputs and outputs
		#if __SHOW_DATASET_
			std::cout<<"Inputs: "<<std::endl;
			for(auto& x : wineNN.getInputs()){
				for(auto& y : x){
					std::cout<<y<<" ";
				}
				std::cout<<std::endl;
			}
			std::cout<<"Outputs: "<<std::endl;
			for(auto& x : wineNN.getOutputs()){
				for(auto& y : x){
					std::cout<<y<<" ";
				}
				std::cout<<std::endl;
			}
		#endif

		std::cout<<"WINE"<<std::endl;
		//run(wineNN,wineInputs,wineOutputs);
		wineNN.testWithOutput();

		#if __SHOW_OUTPUT_PLOT_
			showPlot(wineNN,wineNN.getInputs(),wineNN.getOutputs());
		#endif

		#if _SHOW_LAYERS_
			wineNN.showTopology();
		#endif
	#endif


    return 0;
}
