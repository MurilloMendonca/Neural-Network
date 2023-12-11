#pragma once
#include <hip/hip_runtime.h>

__global__ void forwardPassKernel(const double* input, const double* weightsBase, const double* biasesBase, 
                        double* output, int numInputNeurons, int numOutputNeurons, 
                        size_t weightOffset, size_t biasOffset) ;