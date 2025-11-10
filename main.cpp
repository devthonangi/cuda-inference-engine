#include "inference_engine.hpp"
#include "cuda_kernels.cuh"
#include <iostream>

int main()
{
    std::string model = "models/model.onnx";
    InferenceEngine engine(model);

    const int inputSize = 224 * 224 * 3;
    const int outputSize = 1000;
    float* input = new float[inputSize];
    float* output = new float[outputSize];

    // Run inference
    engine.runInference(input, output);

    // Apply custom CUDA kernel
    float* d_output;
    cudaMalloc(&d_output, outputSize * sizeof(float));
    cudaMemcpy(d_output, output, outputSize * sizeof(float), cudaMemcpyHostToDevice);
    launchReLU(d_output, outputSize);

    std::cout << "Inference completed successfully!" << std::endl;

    cudaFree(d_output);
    delete[] input;
    delete[] output;
    return 0;
}
