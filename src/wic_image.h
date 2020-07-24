#ifndef WIC_IMAGE_H
#define WIC_IMAGE_H

// image decoder and encoder with WIC
#include <wincodec.h>

unsigned char* wic_decode_image(const wchar_t* filepath, int* w, int* h, int* c)
{
    IWICImagingFactory* factory = 0;
    IWICBitmapDecoder* decoder = 0;
    IWICBitmapFrameDecode* frame = 0;
    WICPixelFormatGUID pixel_format;
    IWICFormatConverter* converter = 0;
    IWICBitmap* bitmap = 0;
    IWICBitmapLock* lock = 0;
    int width = 0;
    int height = 0;
    int channels = 0;
    WICRect rect = { 0, 0, 0, 0 };
    unsigned int datasize = 0;
    unsigned char* data = 0;
    int stride = 0;
    unsigned char* bgrdata = 0;

    if (CoCreateInstance(CLSID_WICImagingFactory1, 0, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)))
        goto RETURN;

    if (factory->CreateDecoderFromFilename(filepath, 0, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &decoder))
        goto RETURN;

    if (decoder->GetFrame(0, &frame))
        goto RETURN;

    if (factory->CreateFormatConverter(&converter))
        goto RETURN;

    if (frame->GetPixelFormat(&pixel_format))
        goto RETURN;

    if (!IsEqualGUID(pixel_format, GUID_WICPixelFormat32bppBGRA))
        pixel_format = GUID_WICPixelFormat24bppBGR;

    channels = IsEqualGUID(pixel_format, GUID_WICPixelFormat32bppBGRA) ? 4 : 3;

    if (converter->Initialize(frame, pixel_format, WICBitmapDitherTypeNone, 0, 0.0, WICBitmapPaletteTypeCustom))
        goto RETURN;

    if (factory->CreateBitmapFromSource(converter, WICBitmapCacheOnDemand, &bitmap))
        goto RETURN;

    if (bitmap->GetSize((UINT*)&width, (UINT*)&height))
        goto RETURN;

    rect.Width = width;
    rect.Height = height;
    if (bitmap->Lock(&rect, WICBitmapLockRead, &lock))
        goto RETURN;

    if (lock->GetDataPointer(&datasize, &data))
        goto RETURN;

    if (lock->GetStride((UINT*)&stride))
        goto RETURN;

    bgrdata = (unsigned char*)malloc(width * height * channels);
    if (!bgrdata)
        goto RETURN;

    for (int y = 0; y < height; y++)
    {
        const unsigned char* ptr = data + y * stride;
        unsigned char* bgrptr = bgrdata + y * width * channels;
        memcpy(bgrptr, ptr, width * channels);
    }

    *w = width;
    *h = height;
    *c = channels;

RETURN:
    if (lock) lock->Release();
    if (bitmap) bitmap->Release();
    if (decoder) decoder->Release();
    if (frame) frame->Release();
    if (converter) converter->Release();
    if (factory) factory->Release();

    return bgrdata;
}

int wic_encode_image(const wchar_t* filepath, int w, int h, int c, void* bgrdata)
{
    IWICImagingFactory* factory = 0;
    IWICStream* stream = 0;
    IWICBitmapEncoder* encoder = 0;
    IWICBitmapFrameEncode* frame = 0;
    WICPixelFormatGUID format = c == 4 ? GUID_WICPixelFormat32bppBGRA : GUID_WICPixelFormat24bppBGR;
    int stride = (w * c * 8 + 7) / 8;
    unsigned char* data = 0;
    int ret = 0;

    if (CoCreateInstance(CLSID_WICImagingFactory1, 0, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)))
        goto RETURN;

    if (factory->CreateStream(&stream))
        goto RETURN;

    if (stream->InitializeFromFilename(filepath, GENERIC_WRITE))
        goto RETURN;

    if (factory->CreateEncoder(GUID_ContainerFormatPng, 0, &encoder))
        goto RETURN;

    if (encoder->Initialize(stream, WICBitmapEncoderNoCache))
        goto RETURN;

    if (encoder->CreateNewFrame(&frame, 0))
        goto RETURN;

    if (frame->Initialize(0))
        goto RETURN;

    if (frame->SetSize((UINT)w, (UINT)h))
        goto RETURN;

    if (frame->SetPixelFormat(&format))
        goto RETURN;

    if (!IsEqualGUID(format, c == 4 ? GUID_WICPixelFormat32bppBGRA : GUID_WICPixelFormat24bppBGR))
        goto RETURN;

    data = (unsigned char*)malloc(h * stride);
    if (!data)
        goto RETURN;

    for (int y = 0; y < h; y++)
    {
        const unsigned char* bgrptr = (const unsigned char*)bgrdata + y * w * c;
        unsigned char* ptr = data + y * stride;
        memcpy(ptr, bgrptr, w * c);
    }

    if (frame->WritePixels(h, stride, h * stride, data))
        goto RETURN;

    if (frame->Commit())
        goto RETURN;

    if (encoder->Commit())
        goto RETURN;

    ret = 1;

RETURN:
    if (data) free(data);
    if (encoder) encoder->Release();
    if (frame) frame->Release();
    if (stream) stream->Release();
    if (factory) factory->Release();

    return ret;
}

int wic_encode_jpeg_image(const wchar_t* filepath, int w, int h, int c, void* bgrdata)
{
    // assert c == 3

    IWICImagingFactory* factory = 0;
    IWICStream* stream = 0;
    IWICBitmapEncoder* encoder = 0;
    IWICBitmapFrameEncode* frame = 0;
    IPropertyBag2* propertybag = 0;
    WICPixelFormatGUID format = GUID_WICPixelFormat24bppBGR;
    int stride = (w * c * 8 + 7) / 8;
    unsigned char* data = 0;
    int ret = 0;

    PROPBAG2 option = { 0 };
    option.pstrName = L"ImageQuality";
    VARIANT varValue;
    VariantInit(&varValue);
    varValue.vt = VT_R4;
    varValue.fltVal = 1.0f;

    if (CoCreateInstance(CLSID_WICImagingFactory1, 0, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)))
        goto RETURN;

    if (factory->CreateStream(&stream))
        goto RETURN;

    if (stream->InitializeFromFilename(filepath, GENERIC_WRITE))
        goto RETURN;

    if (factory->CreateEncoder(GUID_ContainerFormatJpeg, 0, &encoder))
        goto RETURN;

    if (encoder->Initialize(stream, WICBitmapEncoderNoCache))
        goto RETURN;

    if (encoder->CreateNewFrame(&frame, &propertybag))
        goto RETURN;

    if (propertybag->Write(1, &option, &varValue))
        goto RETURN;

    if (frame->Initialize(propertybag))
        goto RETURN;

    if (frame->SetSize((UINT)w, (UINT)h))
        goto RETURN;

    if (frame->SetPixelFormat(&format))
        goto RETURN;

    if (!IsEqualGUID(format, GUID_WICPixelFormat24bppBGR))
        goto RETURN;

    data = (unsigned char*)malloc(h * stride);
    if (!data)
        goto RETURN;

    for (int y = 0; y < h; y++)
    {
        const unsigned char* bgrptr = (const unsigned char*)bgrdata + y * w * c;
        unsigned char* ptr = data + y * stride;
        memcpy(ptr, bgrptr, w * c);
    }

    if (frame->WritePixels(h, stride, h * stride, data))
        goto RETURN;

    if (frame->Commit())
        goto RETURN;

    if (encoder->Commit())
        goto RETURN;

    ret = 1;

RETURN:
    if (data) free(data);
    if (encoder) encoder->Release();
    if (frame) frame->Release();
    if (propertybag) propertybag->Release();
    if (stream) stream->Release();
    if (factory) factory->Release();

    return ret;
}

#endif // WIC_IMAGE_H
