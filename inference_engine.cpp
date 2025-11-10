#include "inference_engine.hpp"
#include "cuda_kernels.cuh"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <iostream>

InferenceEngine::InferenceEngine(const std::string& modelPath)
{
    // Deserialize ONNX → TensorRT engine
    runtime = nvinfer1::createInferRuntime(logger);
    auto parser = nvonnxparser::createParser(*network, logger);
    parser->parseFromFile(modelPath.c_str(), 1);

    engine = builder->buildCudaEngine(*network);
    context = engine->createExecutionContext();
}

void InferenceEngine::runInference(float* input, float* output)
{
    void* buffers[2];
    cudaMalloc(&buffers[inputIndex], inputSize * sizeof(float));
    cudaMalloc(&buffers[outputIndex], outputSize * sizeof(float));

    cudaMemcpy(buffers[inputIndex], input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    context->enqueueV2(buffers, stream, nullptr);
    cudaMemcpy(output, buffers[outputIndex], outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
}
