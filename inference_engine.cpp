#include "inference_engine.hpp"
#include "cuda_kernels.cuh"
#include "int8_calibrator.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <chrono>

using namespace nvinfer1;

// Logger for TensorRT
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
} logger;

InferenceEngine::InferenceEngine(const std::string& onnxModel,
                                 PrecisionMode mode,
                                 bool enableInt8,
                                 bool enableFP16)
    : mPrecision(mode), mEnableInt8(enableInt8), mEnableFP16(enableFP16)
{
    std::cout << "\n[INFO] Initializing Inference Engine..." << std::endl;

    // Create builder, network, and parser
    builder = std::unique_ptr<IBuilder>(createInferBuilder(logger));
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));

    if (!parser->parseFromFile(onnxModel.c_str(), static_cast<int>(ILogger::Severity::kINFO))) {
        std::cerr << "[ERROR] Failed to parse ONNX model: " << onnxModel << std::endl;
        exit(1);
    }

    config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(1ULL << 30);  // 1GB

    if (mEnableFP16 && builder->platformHasFastFp16()) {
        config->setFlag(BuilderFlag::kFP16);
        std::cout << "[INFO] FP16 Precision Enabled." << std::endl;
    }

    if (mEnableInt8 && builder->platformHasFastInt8()) {
        config->setFlag(BuilderFlag::kINT8);
        std::vector<std::string> calibrationFiles = {"data/sample1.bin", "data/sample2.bin"};
        auto calibrator = new Int8EntropyCalibrator(calibrationFiles, 224 * 224 * 3, "calibration.cache");
        config->setInt8Calibrator(calibrator);
        std::cout << "[INFO] INT8 Quantization Enabled." << std::endl;
    }

    // Build TensorRT engine
    engine = std::unique_ptr<ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    if (!engine) {
        std::cerr << "[ERROR] Failed to build TensorRT engine!" << std::endl;
        exit(1);
    }

    context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());
    cudaStreamCreate(&mStream);

    std::cout << "[INFO] Inference Engine Initialized Successfully.\n" << std::endl;
}

void InferenceEngine::runInference(std::vector<float>& input, std::vector<float>& output)
{
    nvtxRangePushA("Inference_Run");

    const int inputIndex = engine->getBindingIndex("input");
    const int outputIndex = engine->getBindingIndex("output");

    const size_t inputSize = getSizeByDim(engine->getBindingDimensions(inputIndex));
    const size_t outputSize = getSizeByDim(engine->getBindingDimensions(outputIndex));

    void* buffers[2];
    cudaMalloc(&buffers[inputIndex], inputSize * sizeof(float));
    cudaMalloc(&buffers[outputIndex], outputSize * sizeof(float));

    cudaMemcpyAsync(buffers[inputIndex], input.data(), inputSize * sizeof(float),
                    cudaMemcpyHostToDevice, mStream);

    nvtxRangePushA("TensorRT_Enqueue");
    auto start = std::chrono::high_resolution_clock::now();

    context->enqueueV2(buffers, mStream, nullptr);

    cudaStreamSynchronize(mStream);
    auto end = std::chrono::high_resolution_clock::now();
    nvtxRangePop();

    std::chrono::duration<double, std::milli> inferenceTime = end - start;
    std::cout << "[PROFILE] Inference Time: " << inferenceTime.count() << " ms" << std::endl;

    cudaMemcpyAsync(output.data(), buffers[outputIndex], outputSize * sizeof(float),
                    cudaMemcpyDeviceToHost, mStream);
    cudaStreamSynchronize(mStream);

    // Custom CUDA kernel (optional)
    float* d_output;
    cudaMalloc(&d_output, outputSize * sizeof(float));
    cudaMemcpy(d_output, output.data(), outputSize * sizeof(float), cudaMemcpyHostToDevice);
    launchReLU(d_output, outputSize);
    cudaMemcpy(output.data(), d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_output);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);

    nvtxRangePop();
}

size_t InferenceEngine::getSizeByDim(const Dims& dims)
{
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; i++)
        size *= dims.d[i];
    return size;
}

InferenceEngine::~InferenceEngine()
{
    cudaStreamDestroy(mStream);
    std::cout << "[INFO] Inference Engine destroyed successfully." << std::endl;
}
