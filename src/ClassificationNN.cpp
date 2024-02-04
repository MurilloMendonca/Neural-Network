#include "../include/ClassificationNN.hpp"

ClassificationNN::ClassificationNN(){};

ClassificationNN::ClassificationNN(const topologyType &topology,
                                   std::string activationFunction)
    : NeuralNetwork(topology, activationFunction){};
ClassificationNN::ClassificationNN(const topologyType &topology,
                                   const inputsType &inputs,
                                   const outputsType &outputs,
                                   std::string activationFunction)
    : NeuralNetwork(topology, activationFunction) {
  this->inputs = inputs;
  this->outputs = outputs;
  normalizeInputs();
}

ClassificationNN::ClassificationNN(const topologyType &topology,
                                   const std::string &fileName,
                                   int numberOfCollums, int indexOfClassCollumn,
                                   std::string activationFunction)
    : NeuralNetwork(topology, activationFunction) {
  readDataset(fileName, numberOfCollums, indexOfClassCollumn);
  normalizeInputs();
}

void ClassificationNN::readDataset(const std::string &fileName,
                                   int numberOfCollums, int outputCollum) {
  FileStream fs(fileName);
  std::string word;
  std::map<std::string, double> classes;
  word = fs.getDelimiter(',');

  this->inputs.clear();
  this->outputs.clear();

  while (word != "") {
    std::vector<double> input;
    for (int i = 0; i < numberOfCollums; i++) {
      if (i == outputCollum) {
        if (classes.find(word) == classes.end()) {
          classes[word] = classes.size();
        }
        outputs.push_back({classes[word]});
        word = fs.getDelimiter(',');

        continue;
      }
      input.push_back(word == "?" ? 0.0 : std::stod(word));
      word = fs.getDelimiter(',');
    }
    inputs.push_back(input);
  }

  for (auto &x : outputs) {
    int classification = x[0];
    x.clear();
    for (int i = 0; i < classes.size(); i++) {
      if (i == classification) {
        x.push_back(1);
      } else {
        x.push_back(0);
      }
    }
  }
#if _VERBOSE_
  std::cout << "Classes: " << std::endl;
  for (auto it = classes.begin(); it != classes.end(); it++) {
    std::cout << it->first << " " << it->second << std::endl;
  }
#endif
}

void ClassificationNN::normalizeInputs() {
  for (int i = 0; i < inputs[0].size(); i++) {
    double max = inputs[0][i];
    double min = inputs[0][i];
    for (int j = 0; j < inputs.size(); j++) {
      if (inputs[j][i] > max) {
        max = inputs[j][i];
      }
      if (inputs[j][i] < min) {
        min = inputs[j][i];
      }
    }
    for (int j = 0; j < inputs.size(); j++) {
      inputs[j][i] = (inputs[j][i] - min) / (max - min);
    }
  }
}

void ClassificationNN::train(int epochs) {
  for (int i = 0; i < epochs; i++) {
    for (int j = 0; j < inputs.size(); j++) {
      forward(inputs[j]);
      backpropagate(outputs[j]);
    }
  }
}

float ClassificationNN::test() {
  int errorNumber = 0;
  std::vector<int> truePositives(
      outputs[0].size(), 0); // Initialize true positives for each class to 0
  std::vector<int> falsePositives(
      outputs[0].size(), 0); // Initialize false positives for each class to 0

  for (size_t i = 0; i < inputs.size(); ++i) {
    this->forward(inputs[i]);
    const auto &resultLayer = this->getLayers().back().neurons;

    int classification = 0;
    for (size_t j = 0; j < outputs[0].size(); j++) {
      if (resultLayer[j].value > resultLayer[classification].value) {
        classification = j;
      }
    }
    int expectedClassification = 0;
    for (size_t j = 0; j < outputs[0].size(); j++) {
      if (outputs[i][j] > outputs[i][expectedClassification]) {
        expectedClassification = j;
      }
    }
    if (expectedClassification == classification) {
      truePositives[classification]++;
    } else {
      falsePositives[classification]++;
      errorNumber++;
    }
  }

  return (double)(inputs.size() - errorNumber) / inputs.size();
}

void ClassificationNN::testWithOutput() {
  double totalError = 0.0;
  int errorNumber = 0;
  std::vector<int> truePositives(
      outputs[0].size(), 0); // Initialize true positives for each class to 0
  std::vector<int> falsePositives(
      outputs[0].size(), 0); // Initialize false positives for each class to 0

  for (size_t i = 0; i < inputs.size(); ++i) {
    forward(inputs[i]);
    const auto &resultLayer = getLayers().back().neurons;

    double exampleError = 0.0;
    int classification = 0;
    for (size_t j = 0; j < outputs[0].size(); j++) {
      double error = outputs[i][j] - resultLayer[j].value;
      exampleError += error * error; // squared error
      if (resultLayer[j].value > resultLayer[classification].value) {
        classification = j;
      }
    }
    int expectedClassification = 0;
    for (size_t j = 0; j < outputs[0].size(); j++) {
      if (outputs[i][j] > outputs[i][expectedClassification]) {
        expectedClassification = j;
      }
    }
    if (expectedClassification == classification) {
      truePositives[classification]++;
    } else {
      falsePositives[classification]++;
      errorNumber++;
    }
#if _VERBOSE_
    std::cout << "Input: ";
    for (auto val : inputs[i])
      std::cout << val << " ";
    std::cout << "| Expected output: " << expectedClassification
              << " | Network output: " << classification << std::endl;
#endif
    totalError +=
        exampleError / outputs[0].size(); // average error for this example
  }

  double mse =
      totalError / inputs.size(); // mean squared error over all examples
  std::cout << "Mean Squared Error (MSE) on All Data: " << mse << std::endl;

  std::cout << "Wrong Predictions: " << errorNumber << std::endl;
  std::cout << "Right Predictions: " << inputs.size() - errorNumber
            << std::endl;
  std::cout << "Accuracy: "
            << (double)(inputs.size() - errorNumber) / inputs.size() * 100
            << "%" << std::endl;

  // Compute and display precision for each class
  for (size_t i = 0; i < outputs[0].size(); i++) {
    double precision = static_cast<double>(truePositives[i]) /
                       (truePositives[i] + falsePositives[i]);
    std::cout << "Precision for class " << i << ": " << precision * 100 << "%"
              << std::endl;
  }
}

void ClassificationNN::trainAndTest(int epochs = 1000) {
  // Shuffle the dataset
  auto shuffled_indices = std::vector<size_t>(inputs.size());
  std::iota(shuffled_indices.begin(), shuffled_indices.end(),
            0); // Fill with 0, 1, ..., n-1
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle(shuffled_indices.begin(), shuffled_indices.end(),
               std::default_random_engine(seed));

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
      forward(trainingInputs[i]);
      backpropagate(
          trainingOutputs[i]); // Assuming your NN has a backprop method
    }

    // Test on test data and compute MSE
#if _VERBOSE_
    double totalError = 0.0;
    for (size_t i = 0; i < testInputs.size(); ++i) {
      forward(testInputs[i]);
      const auto &resultLayer = getLayers().back().neurons;

      double exampleError = 0.0;
      for (size_t j = 0; j < testOutputs[0].size(); j++) {
        double error = testOutputs[i][j] - resultLayer[j].value;
        exampleError += error * error; // squared error
      }
      totalError += exampleError /
                    testOutputs[0].size(); // average error for this example
    }
    double mse = totalError / testInputs.size();
    std::cout << "Epoch " << (epoch + 1)
              << " - Mean Squared Error (MSE) on Test Data: " << mse
              << std::endl;
#endif
  }
}

void ClassificationNN::showTopology() {
  std::cout << "Network topology:" << std::endl;
  int layerNum = 1; // For display purposes
  for (const auto &layer : getLayers()) {
    std::cout << "Layer " << layerNum << " (" << layer.neurons.size()
              << " neurons):" << std::endl;
    int neuronNum = 1; // For display purposes
    for (const auto &neuron : layer.neurons) {
      std::cout << "\tNeuron " << neuronNum << " weights: ";
      for (const auto &weight : neuron.weights) {
        std::cout << weight << " ";
      }
      std::cout << "| Bias: " << neuron.bias
                << std::endl; // Also displaying the bias for completeness
      neuronNum++;
    }
    layerNum++;
  }
  std::cout << std::endl;
}

void ClassificationNN::setInputs(const inputsType &inputs) {
  this->inputs = inputs;
}
void ClassificationNN::setOutputs(const outputsType &outputs) {
  this->outputs = outputs;
}
inputsType ClassificationNN::getInputs() const { return inputs; }

outputsType ClassificationNN::getOutputs() const { return outputs; }
