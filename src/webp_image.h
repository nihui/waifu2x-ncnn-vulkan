#ifndef WEBP_IMAGE_H
#define WEBP_IMAGE_H

// webp image decoder and encoder with libwebp
#include <stdio.h>
#include <stdlib.h>
#include "webp/decode.h"
#include "webp/encode.h"

unsigned char* webp_load(const unsigned char* buffer, int len, int* w, int* h, int* c)
{
    unsigned char* pixeldata = 0;

    WebPDecoderConfig config;
    WebPInitDecoderConfig(&config);

    if (WebPGetFeatures(buffer, len, &config.input) != VP8_STATUS_OK)
        return NULL;

    int width = config.input.width;
    int height = config.input.height;
    int channels = config.input.has_alpha ? 4 : 3;

    pixeldata = (unsigned char*)malloc(width * height * channels);

#if _WIN32
    config.output.colorspace = channels == 4 ? MODE_BGRA : MODE_BGR;
#else
    config.output.colorspace = channels == 4 ? MODE_RGBA : MODE_RGB;
#endif

    config.output.u.RGBA.stride = width * channels;
    config.output.u.RGBA.size = width * height * channels;
    config.output.u.RGBA.rgba = pixeldata;
    config.output.is_external_memory = 1;

    if (WebPDecode(buffer, len, &config) != VP8_STATUS_OK)
    {
        free(pixeldata);
        return NULL;
    }

    *w = width;
    *h = height;
    *c = channels;

    return pixeldata;
}

#if _WIN32
int webp_save(const wchar_t* filepath, int w, int h, int c, const unsigned char* pixeldata)
#else
int webp_save(const char* filepath, int w, int h, int c, const unsigned char* pixeldata)
#endif
{
    int ret = 0;

    unsigned char* output = 0;
    size_t length = 0;

    FILE* fp = 0;

    if (c == 3)
    {
#if _WIN32
        length = WebPEncodeLosslessBGR(pixeldata, w, h, w * 3, &output);
#else
        length = WebPEncodeLosslessRGB(pixeldata, w, h, w * 3, &output);
#endif
    }
    else if (c == 4)
    {
#if _WIN32
        length = WebPEncodeLosslessBGRA(pixeldata, w, h, w * 4, &output);
#else
        length = WebPEncodeLosslessRGBA(pixeldata, w, h, w * 4, &output);
#endif
    }
    else
    {
        // unsupported channel type
    }

    if (length == 0)
        goto RETURN;

#if _WIN32
    fp = _wfopen(filepath, L"wb");
#else
    fp = fopen(filepath, "wb");
#endif
    if (!fp)
        goto RETURN;

    fwrite(output, 1, length, fp);

    ret = 1;

RETURN:
    if (output) WebPFree(output);
    if (fp) fclose(fp);

    return ret;
}

#endif // WEBP_IMAGE_H
