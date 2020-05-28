#ifndef WEBP_IMAGE_H
#define WEBP_IMAGE_H

// webp image decoder with libwebp
#include <stdio.h>
#include <stdlib.h>
#include "webp/decode.h"

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

#endif // WEBP_IMAGE_H
