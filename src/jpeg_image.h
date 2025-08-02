#ifndef JPEG_IMAGE_H
#define JPEG_IMAGE_H

// jpeg image decoder and encoder with libjpeg
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include "jpeglib.h"

struct safe_jpeg_error_mgr
{
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

static void safe_jpeg_error_exit(j_common_ptr cinfo)
{
    struct safe_jpeg_error_mgr* myerr = (struct safe_jpeg_error_mgr*)(cinfo->err);
    longjmp(myerr->setjmp_buffer, 1);
}

static void safe_jpeg_emit_message(j_common_ptr cinfo, int msg_level)
{
}

static void safe_jpeg_output_message(j_common_ptr cinfo)
{
}

unsigned char* jpeg_load(const unsigned char* buffer, int len, int* w, int* h, int* c)
{
    unsigned char* pixeldata = 0;

    struct jpeg_decompress_struct cinfo;
    struct safe_jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr.pub);
    cinfo.err->error_exit = safe_jpeg_error_exit;
    cinfo.err->emit_message = safe_jpeg_emit_message;
    cinfo.err->output_message = safe_jpeg_output_message;

    if (setjmp(jerr.setjmp_buffer))
    {
        jpeg_destroy_decompress(&cinfo);
        return NULL;
    }

    jpeg_create_decompress(&cinfo);

    jpeg_mem_src(&cinfo, (unsigned char*)buffer, len);

    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK)
    {
        jpeg_destroy_decompress(&cinfo);
        return NULL;
    }

    // always output RGB 8bit
    cinfo.out_color_space = JCS_RGB;
    cinfo.output_components = 3;
    cinfo.data_precision = 8;

    jpeg_start_decompress(&cinfo);

    int width = cinfo.image_width;
    int height = cinfo.image_height;

    pixeldata = (unsigned char*)malloc(width * height * 3);

    while (cinfo.output_scanline < cinfo.output_height)
    {
        unsigned char* ptr = (unsigned char*)pixeldata + (size_t)cinfo.output_scanline * width * 3;
        jpeg_read_scanlines(&cinfo, &ptr, 1);
    }

    jpeg_finish_decompress(&cinfo);

    jpeg_destroy_decompress(&cinfo);

    *w = width;
    *h = height;
    *c = 3;

    return pixeldata;
}

#if _WIN32
int jpeg_save(const wchar_t* filepath, int w, int h, int c, const unsigned char* pixeldata)
#else
int jpeg_save(const char* filepath, int w, int h, int c, const unsigned char* pixeldata)
#endif
{
    struct jpeg_compress_struct cinfo;
    struct safe_jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr.pub);
    cinfo.err->error_exit = safe_jpeg_error_exit;
    cinfo.err->emit_message = safe_jpeg_emit_message;
    cinfo.err->output_message = safe_jpeg_output_message;

    if (setjmp(jerr.setjmp_buffer))
    {
        jpeg_destroy_compress(&cinfo);
        return 0;
    }

#if _WIN32
    FILE* fp = _wfopen(filepath, L"wb");
#else
    FILE* fp = fopen(filepath, "wb");
#endif
    if (!fp)
    {
        jpeg_destroy_compress(&cinfo);
        return 0;
    }

    jpeg_create_compress(&cinfo);

    jpeg_stdio_dest(&cinfo, fp);

    cinfo.image_width = w;
    cinfo.image_height = h;

    // always input RGB 8bit
    cinfo.in_color_space = JCS_RGB;
    cinfo.input_components = 3;
    cinfo.data_precision = 8;

    jpeg_set_defaults(&cinfo);

    // no downsampling
    cinfo.comp_info[0].h_samp_factor = 1;
    cinfo.comp_info[0].v_samp_factor = 1;
    cinfo.comp_info[1].h_samp_factor = 1;
    cinfo.comp_info[1].v_samp_factor = 1;
    cinfo.comp_info[2].h_samp_factor = 1;
    cinfo.comp_info[2].v_samp_factor = 1;

    jpeg_set_quality(&cinfo, 98, TRUE);

    jpeg_start_compress(&cinfo, TRUE);

    for (int y = 0; y < h; y++)
    {
        const unsigned char* ptr = pixeldata + (size_t)y * w * 3;
        jpeg_write_scanlines(&cinfo, (unsigned char**)&ptr, 1);
    }

    jpeg_finish_compress(&cinfo);

    jpeg_destroy_compress(&cinfo);

    fclose(fp);

    return 1;
}

#endif // JPEG_IMAGE_H
