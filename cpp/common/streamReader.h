#pragma once

#include <NvInferRuntime.h>
#include <fstream>
#include <string>
#include <iostream>

namespace trt_edgellm
{
namespace file_io
{

class FileStreamReader : public nvinfer1::IStreamReader
{
public:
    FileStreamReader(std::string const& path)
        : mFile(path, std::ios::binary)
    {
        if (!mFile.is_open())
        {
            throw std::runtime_error("Failed to open file for streaming: " + path);
        }
    }

    ~FileStreamReader() override = default;

    int64_t read(void* destination, int64_t nbBytes) override
    {
        if (!mFile.good() || mFile.eof())
        {
            return 0;
        }

        mFile.read(static_cast<char*>(destination), nbBytes);
        return mFile.gcount();
    }

private:
    std::ifstream mFile;
};

} // namespace file_io
} // namespace trt_edgellm
