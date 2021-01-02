// waifu2x implemented with ncnn library

#ifndef WAIFU2X_H
#define WAIFU2X_H

#include <string>

// ncnn
#include "net.h"
#include "gpu.h"
#include "layer.h"

class Waifu2x
{
public:
    Waifu2x(int gpuid, bool tta_mode = false, int num_threads = 1);
    ~Waifu2x();

#if _WIN32
    int load(const std::wstring& parampath, const std::wstring& modelpath);
#else
    int load(const std::string& parampath, const std::string& modelpath);
#endif

    int process(const ncnn::Mat& inimage, ncnn::Mat& outimage) const;

    int process_cpu(const ncnn::Mat& inimage, ncnn::Mat& outimage) const;

public:
    // waifu2x parameters
    int noise;
    int scale;
    int tilesize;
    int prepadding;

private:
    ncnn::VulkanDevice* vkdev;
    ncnn::Net net;
    ncnn::Pipeline* waifu2x_preproc;
    ncnn::Pipeline* waifu2x_postproc;
    ncnn::Layer* bicubic_2x;
    bool tta_mode;
};

#endif // WAIFU2X_H
