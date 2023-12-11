#include "Kernels.hpp"


__global__ void forwardPassKernel(const double* input, const double* weightsBase, const double* biasesBase, 
                                double* output, int numInputNeurons, int numOutputNeurons, size_t weightOffset, 
                                size_t biasOffset) 
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < numOutputNeurons) {
        // Apply the offsets to get the correct weights and biases for the current layer
        size_t weightStartIndex = weightOffset + id * numInputNeurons;

        const double* weights = weightsBase + weightStartIndex;
        const double* biases = biasesBase + biasOffset;

        double sum = 0.0;
        for (int i = 0; i < numInputNeurons; ++i) {
            sum += input[i] * weights[i];
            //printf("input[%d] = %f, weights[%d] = %f\n", i, input[i], i, weights[i]);
        }
        output[id] = 1.0 / (1.0 + exp(-(sum+biases[id])));
        //printf("output[%d] = %f\n", id, output[id]);
    }
}

