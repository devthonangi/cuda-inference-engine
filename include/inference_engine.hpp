#pragma once
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <memory>
#include <vector>
#include <string>

enum class PrecisionMode { FP32, FP16, INT8 };

class InferenceEngine {
public:
    InferenceEngine(const std::string& onnxModel,
                    PrecisionMode mode = PrecisionMode::FP32,
                    bool enableInt8 = false,
                    bool enableFP16 = false);
    ~InferenceEngine();

    void runInference(std::vector<float>& input, std::vector<float>& output);

private:
    size_t getSizeByDim(const nvinfer1::Dims& dims);

    std::unique_ptr<nvinfer1::IBuilder> builder;
    std::unique_ptr<nvinfer1::INetworkDefinition> network;
    std::unique_ptr<nvonnxparser::IParser> parser;
    std::unique_ptr<nvinfer1::IBuilderConfig> config;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    cudaStream_t mStream{};
    PrecisionMode mPrecision;
    bool mEnableInt8;
    bool mEnableFP16;
};
