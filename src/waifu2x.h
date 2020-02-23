// waifu2x implemented with ncnn library

#ifndef WAIFU2X_H
#define WAIFU2X_H

#include <string>

// ncnn
#include "net.h"
#include "gpu.h"

class Waifu2x
{
public:
    Waifu2x(int gpuid, bool tta_mode = false);
    ~Waifu2x();

#if _WIN32
    int load(const std::wstring& parampath, const std::wstring& modelpath);
#else
    int load(const std::string& parampath, const std::string& modelpath);
#endif

    int process(const ncnn::Mat& inimage, ncnn::Mat& outimage) const;

public:
    // waifu2x parameters
    int noise;
    int scale;
    int tilesize;
    int prepadding;

private:
    ncnn::Net net;
    ncnn::Pipeline* waifu2x_preproc;
    ncnn::Pipeline* waifu2x_postproc;
    bool tta_mode;
};

#endif // WAIFU2X_H
