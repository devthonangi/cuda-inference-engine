#pragma once
#include <NvInfer.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator(const std::vector<std::string>& calibrationFiles,
                          int inputSize,
                          const std::string& cacheFile)
        : mCalibrationFiles(calibrationFiles),
          mInputSize(inputSize),
          mCacheFile(cacheFile),
          mFileIndex(0)
    {
        cudaMalloc(&mDeviceInput, mInputSize * sizeof(float));
    }

    ~Int8EntropyCalibrator() override {
        cudaFree(mDeviceInput);
    }

    int getBatchSize() const noexcept override { return 1; }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override {
        if (mFileIndex >= mCalibrationFiles.size()) return false;

        std::ifstream file(mCalibrationFiles[mFileIndex], std::ios::binary);
        std::vector<float> input(mInputSize);
        file.read(reinterpret_cast<char*>(input.data()), mInputSize * sizeof(float));
        cudaMemcpy(mDeviceInput, input.data(), mInputSize * sizeof(float), cudaMemcpyHostToDevice);
        bindings[0] = mDeviceInput;
        mFileIndex++;
        return true;
    }

    const void* readCalibrationCache(size_t& length) noexcept override {
        mCalibrationCache.clear();
        std::ifstream input(mCacheFile, std::ios::binary);
        if (input.good()) {
            input.seekg(0, std::ios::end);
            size_t size = input.tellg();
            input.seekg(0, std::ios::beg);
            mCalibrationCache.resize(size);
            input.read(reinterpret_cast<char*>(mCalibrationCache.data()), size);
        }
        length = mCalibrationCache.size();
        return length ? mCalibrationCache.data() : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept override {
        std::ofstream output(mCacheFile, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    std::vector<std::string> mCalibrationFiles;
    int mInputSize;
    std::string mCacheFile;
    int mFileIndex;
    void* mDeviceInput;
    std::vector<char> mCalibrationCache;
};
