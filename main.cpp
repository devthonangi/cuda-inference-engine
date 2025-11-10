#include "inference_engine.hpp"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>

// Utility: simple argument parser
struct Args {
    bool fp16 = false;
    bool int8 = false;
    bool profile = false;
    std::string modelPath = "models/model.onnx";
};

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--fp16") == 0) args.fp16 = true;
        else if (strcmp(argv[i], "--int8") == 0) args.int8 = true;
        else if (strcmp(argv[i], "--profile") == 0) args.profile = true;
        else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) args.modelPath = argv[++i];
        else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: ./cuda_inference_engine [--fp16] [--int8] [--profile] [--model path/to/model.onnx]\n";
            exit(0);
        }
    }
    return args;
}

int main(int argc, char** argv)
{
    std::cout << "\n==============================\n";
    std::cout << " CUDA-Accelerated Deep Learning Inference Engine\n";
    std::cout << "==============================\n";

    Args args = parseArgs(argc, argv);

    // Display configuration summary
    std::cout << "\n[CONFIGURATION]\n";
    std::cout << "Model:    " << args.modelPath << "\n";
    std::cout << "FP16:     " << (args.fp16 ? "Enabled" : "Disabled") << "\n";
    std::cout << "INT8:     " << (args.int8 ? "Enabled" : "Disabled") << "\n";
    std::cout << "Profiling:" << (args.profile ? "Enabled" : "Disabled") << "\n\n";

    try {
        // Initialize engine
        PrecisionMode mode = PrecisionMode::FP32;
        if (args.fp16) mode = PrecisionMode::FP16;
        if (args.int8) mode = PrecisionMode::INT8;

        nvtxRangePushA("Engine_Initialization");
        InferenceEngine engine(args.modelPath, mode, args.int8, args.fp16);
        nvtxRangePop();

        // Create random input
        const int inputSize = 224 * 224 * 3;  // Example shape (HWC)
        const int outputSize = 1000;          // Example number of classes

        std::vector<float> input(inputSize, 0.5f);
        std::vector<float> output(outputSize, 0.0f);

        if (args.profile) {
            nvtxRangePushA("Profiling_Run");
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        engine.runInference(input, output);
        auto t2 = std::chrono::high_resolution_clock::now();

        if (args.profile) {
            nvtxRangePop();
        }

        std::chrono::duration<double, std::milli> totalTime = t2 - t1;
        std::cout << "[PROFILE] Total Execution Time: " << totalTime.count() << " ms\n";

        // Display output summary
        std::cout << "\n[OUTPUT SAMPLE]\n";
        for (int i = 0; i < std::min(10, (int)output.size()); ++i)
            std::cout << "  output[" << i << "] = " << output[i] << std::endl;

        std::cout << "\n Inference completed successfully.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "[EXCEPTION] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
