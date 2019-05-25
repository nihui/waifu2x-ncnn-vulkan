// waifu2x implemented with ncnn library

#include <stdio.h>
#include <algorithm>
#include <vector>

#if WIN32
// image decoder and encoder with wic
#include "wic_image.h"
#else // WIN32
// image decoder and encoder with stb
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_PIC
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif // WIN32

// ncnn
#include "layer_type.h"
#include "net.h"
#include "gpu.h"

static const uint32_t waifu2x_preproc_spv_data[] = {
    #include "waifu2x_preproc.spv.hex.h"
};
static const uint32_t waifu2x_preproc_fp16s_spv_data[] = {
    #include "waifu2x_preproc_fp16s.spv.hex.h"
};
static const uint32_t waifu2x_preproc_int8s_spv_data[] = {
    #include "waifu2x_preproc_int8s.spv.hex.h"
};
static const uint32_t waifu2x_postproc_spv_data[] = {
    #include "waifu2x_postproc.spv.hex.h"
};
static const uint32_t waifu2x_postproc_fp16s_spv_data[] = {
    #include "waifu2x_postproc_fp16s.spv.hex.h"
};
static const uint32_t waifu2x_postproc_int8s_spv_data[] = {
    #include "waifu2x_postproc_int8s.spv.hex.h"
};

#if WIN32
int wmain(int argc, wchar_t** argv)
#else
int main(int argc, char** argv)
#endif
{
#if WIN32
    if (argc != 5 && argc != 6)
    {
        fprintf(stderr, "Usage: %ls [image] [outputpng] [noise=-1/0/1/2/3] [scale=1/2] [tilesize=400]\n", argv[0]);
        return -1;
    }

    const wchar_t* imagepath = argv[1];
    const wchar_t* outputpngpath = argv[2];
    int noise = _wtoi(argv[3]);
    int scale = _wtoi(argv[4]);
    int tilesize = argc == 6 ? _wtoi(argv[5]) : 400;
#else
    if (argc != 5 && argc != 6)
    {
        fprintf(stderr, "Usage: %s [image] [outputpng] [noise=-1/0/1/2/3] [scale=1/2] [tilesize=400]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];
    const char* outputpngpath = argv[2];
    int noise = atoi(argv[3]);
    int scale = atoi(argv[4]);
    int tilesize = argc == 6 ? atoi(argv[5]) : 400;
#endif

    if (noise < -1 || noise > 3 || scale < 1 || scale > 2)
    {
        fprintf(stderr, "invalid noise or scale argument\n");
        return -1;
    }

    if (tilesize < 32)
    {
        fprintf(stderr, "invalid tilesize argument\n");
        return -1;
    }

    const int TILE_SIZE_X = tilesize;
    const int TILE_SIZE_Y = tilesize;

    int prepadding = 0;
    char parampath[256];
    char modelpath[256];
    if (noise == -1)
    {
        prepadding = 18;
        sprintf(parampath, "models-cunet/scale2.0x_model.param");
        sprintf(modelpath, "models-cunet/scale2.0x_model.bin");
    }
    else if (scale == 1)
    {
        prepadding = 28;
        sprintf(parampath, "models-cunet/noise%d_model.param", noise);
        sprintf(modelpath, "models-cunet/noise%d_model.bin", noise);
    }
    else if (scale == 2)
    {
        prepadding = 18;
        sprintf(parampath, "models-cunet/noise%d_scale2.0x_model.param", noise);
        sprintf(modelpath, "models-cunet/noise%d_scale2.0x_model.bin", noise);
    }

#if WIN32
    CoInitialize(0);
#endif

    ncnn::create_gpu_instance();

    ncnn::VulkanDevice* vkdev = new ncnn::VulkanDevice;

    // HACK ncnn fp16a produce incorrect result, force off
    // TODO provide a way to control storage and arothmetic precision in ncnn
    ((ncnn::GpuInfo*)(&vkdev->info))->support_fp16_arithmetic = false;

    {
        ncnn::Net waifu2x;

        waifu2x.use_vulkan_compute = true;
        waifu2x.set_vulkan_device(vkdev);

        waifu2x.load_param(parampath);
        waifu2x.load_model(modelpath);

        ncnn::Option opt = ncnn::get_default_option();
        opt.blob_vkallocator = vkdev->allocator();
        opt.workspace_vkallocator = vkdev->allocator();
        opt.staging_vkallocator = vkdev->staging_allocator();

        // initialize preprocess and postprocess pipeline
        ncnn::Pipeline* waifu2x_preproc;
        ncnn::Pipeline* waifu2x_postproc;
        {
            std::vector<ncnn::vk_specialization_type> specializations(1);
#if WIN32
            specializations[0].i = 1;
#else
            specializations[0].i = 0;
#endif

            waifu2x_preproc = new ncnn::Pipeline(vkdev);
            waifu2x_preproc->set_optimal_local_size_xyz(32, 32, 3);
            if (vkdev->info.support_fp16_storage && vkdev->info.support_int8_storage)
                waifu2x_preproc->create(waifu2x_preproc_int8s_spv_data, sizeof(waifu2x_preproc_int8s_spv_data), "waifu2x_preproc_int8s", specializations, 2, 9);
            else if (vkdev->info.support_fp16_storage)
                waifu2x_preproc->create(waifu2x_preproc_fp16s_spv_data, sizeof(waifu2x_preproc_fp16s_spv_data), "waifu2x_preproc_fp16s", specializations, 2, 9);
            else
                waifu2x_preproc->create(waifu2x_preproc_spv_data, sizeof(waifu2x_preproc_spv_data), "waifu2x_preproc", specializations, 2, 9);

            waifu2x_postproc = new ncnn::Pipeline(vkdev);
            waifu2x_postproc->set_optimal_local_size_xyz(32, 32, 3);
            if (vkdev->info.support_fp16_storage && vkdev->info.support_int8_storage)
                waifu2x_postproc->create(waifu2x_postproc_int8s_spv_data, sizeof(waifu2x_postproc_int8s_spv_data), "waifu2x_postproc_int8s", specializations, 2, 8);
            else if (vkdev->info.support_fp16_storage)
                waifu2x_postproc->create(waifu2x_postproc_fp16s_spv_data, sizeof(waifu2x_postproc_fp16s_spv_data), "waifu2x_postproc_fp16s", specializations, 2, 8);
            else
                waifu2x_postproc->create(waifu2x_postproc_spv_data, sizeof(waifu2x_postproc_spv_data), "waifu2x_postproc", specializations, 2, 8);
        }

        // main routine
        {
#if WIN32
            int w, h, c;
            unsigned char* bgrdata = wic_decode_image(imagepath, &w, &h, &c);
            if (!bgrdata)
            {
                fprintf(stderr, "decode image %ls failed\n", imagepath);
                return -1;
            }

            ncnn::Mat outbgr(w * scale, h * scale, (size_t)3u, 3);
#else // WIN32
            int w, h, c;
            unsigned char* rgbdata = stbi_load(imagepath, &w, &h, &c, 3);
            if (!rgbdata)
            {
                fprintf(stderr, "decode image %s failed\n", imagepath);
                return -1;
            }

            ncnn::Mat outrgb(w * scale, h * scale, (size_t)3u, 3);
#endif // WIN32

            // prepadding
            int prepadding_bottom = prepadding;
            int prepadding_right = prepadding;
            if (scale == 1)
            {
                prepadding_bottom += (h + 3) / 4 * 4 - h;
                prepadding_right += (w + 3) / 4 * 4 - w;
            }
            if (scale == 2)
            {
                prepadding_bottom += (h + 1) / 2 * 2 - h;
                prepadding_right += (w + 1) / 2 * 2 - w;
            }

            // each tile 400x400
            int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
            int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

            // TODO #pragma omp parallel for
            for (int yi = 0; yi < ytiles; yi++)
            {
                int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
                int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);

                ncnn::Mat in;
                if (vkdev->info.support_fp16_storage && vkdev->info.support_int8_storage)
                {
#if WIN32
                    in = ncnn::Mat(w, (in_tile_y1 - in_tile_y0), bgrdata + in_tile_y0 * w * 3, (size_t)3u, 1);
#else
                    in = ncnn::Mat(w, (in_tile_y1 - in_tile_y0), rgbdata + in_tile_y0 * w * 3, (size_t)3u, 1);
#endif
                }
                else
                {
#if WIN32
                    in = ncnn::Mat::from_pixels(bgrdata + in_tile_y0 * w * 3, ncnn::Mat::PIXEL_BGR2RGB, w, (in_tile_y1 - in_tile_y0));
#else
                    in = ncnn::Mat::from_pixels(rgbdata + in_tile_y0 * w * 3, ncnn::Mat::PIXEL_RGB, w, (in_tile_y1 - in_tile_y0));
#endif
                }

                ncnn::VkCompute cmd(vkdev);

                // upload
                ncnn::VkMat in_gpu;
                {
                    in_gpu.create_like(in, opt.blob_vkallocator, opt.staging_vkallocator);

                    in_gpu.prepare_staging_buffer();
                    in_gpu.upload(in);

                    cmd.record_upload(in_gpu);

                    if (xtiles > 1)
                    {
                        cmd.submit_and_wait();
                        cmd.reset();
                    }
                }

                int out_tile_y0 = std::max(yi * TILE_SIZE_Y, 0);
                int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h);

                ncnn::VkMat out_gpu;
                if (vkdev->info.support_fp16_storage && vkdev->info.support_int8_storage)
                {
                    out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, (size_t)3u, 1, opt.blob_vkallocator, opt.staging_vkallocator);
                }
                else
                {
                    out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, 3, (size_t)4u, 1, opt.blob_vkallocator, opt.staging_vkallocator);
                }

                for (int xi = 0; xi < xtiles; xi++)
                {
                    // preproc
                    ncnn::VkMat in_tile_gpu;
                    {
                        // crop tile
                        int tile_x0 = xi * TILE_SIZE_X;
                        int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding + prepadding_right;
                        int tile_y0 = yi * TILE_SIZE_Y;
                        int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding + prepadding_bottom;

                        in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, (size_t)4u, 1, opt.blob_vkallocator, opt.staging_vkallocator);

                        std::vector<ncnn::VkMat> bindings(2);
                        bindings[0] = in_gpu;
                        bindings[1] = in_tile_gpu;

                        std::vector<ncnn::vk_constant_type> constants(9);
                        constants[0].i = in_gpu.w;
                        constants[1].i = in_gpu.h;
                        constants[2].i = in_gpu.cstep;
                        constants[3].i = in_tile_gpu.w;
                        constants[4].i = in_tile_gpu.h;
                        constants[5].i = in_tile_gpu.cstep;
                        constants[6].i = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                        constants[7].i = prepadding;
                        constants[8].i = xi * TILE_SIZE_X;

                        cmd.record_pipeline(waifu2x_preproc, bindings, constants, in_tile_gpu);
                    }

                    // waifu2x
                    ncnn::VkMat out_tile_gpu;
                    {
                        ncnn::Extractor ex = waifu2x.create_extractor();
                        ex.input("Input1", in_tile_gpu);

                        ex.extract("Eltwise4", out_tile_gpu, cmd);
                    }

                    // postproc
                    {
                        std::vector<ncnn::VkMat> bindings(2);
                        bindings[0] = out_tile_gpu;
                        bindings[1] = out_gpu;

                        std::vector<ncnn::vk_constant_type> constants(8);
                        constants[0].i = out_tile_gpu.w;
                        constants[1].i = out_tile_gpu.h;
                        constants[2].i = out_tile_gpu.cstep;
                        constants[3].i = out_gpu.w;
                        constants[4].i = out_gpu.h;
                        constants[5].i = out_gpu.cstep;
                        constants[6].i = xi * TILE_SIZE_X * scale;
                        constants[7].i = out_gpu.w - xi * TILE_SIZE_X * scale;

                        ncnn::VkMat dispatcher;
                        dispatcher.w = out_gpu.w - xi * TILE_SIZE_X * scale;
                        dispatcher.h = out_gpu.h;
                        dispatcher.c = 3;

                        cmd.record_pipeline(waifu2x_postproc, bindings, constants, dispatcher);
                    }

                    if (xtiles > 1)
                    {
                        cmd.submit_and_wait();
                        cmd.reset();
                    }
                }

                // download
                {
                    out_gpu.prepare_staging_buffer();
                    cmd.record_download(out_gpu);

                    cmd.submit_and_wait();
                }

                if (vkdev->info.support_fp16_storage && vkdev->info.support_int8_storage)
                {
#if WIN32
                    ncnn::Mat out(out_gpu.w, out_gpu.h, (unsigned char*)outbgr.data + yi * scale * TILE_SIZE_Y * w * scale * 3, (size_t)3u, 1);
#else
                    ncnn::Mat out(out_gpu.w, out_gpu.h, (unsigned char*)outrgb.data + yi * scale * TILE_SIZE_Y * w * scale * 3, (size_t)3u, 1);
#endif

                    out_gpu.download(out);
                }
                else
                {
                    ncnn::Mat out;
                    out.create_like(out_gpu, opt.blob_allocator);
                    out_gpu.download(out);

#if WIN32
                    out.to_pixels((unsigned char*)outbgr.data + yi * scale * TILE_SIZE_Y * w * scale * 3, ncnn::Mat::PIXEL_RGB2BGR);
#else
                    out.to_pixels((unsigned char*)outrgb.data + yi * scale * TILE_SIZE_Y * w * scale * 3, ncnn::Mat::PIXEL_RGB);
#endif
                }
            }

#if WIN32
            free(bgrdata);

            int ret = wic_encode_image(outputpngpath, outbgr.w, outbgr.h, 3, outbgr.data);
            if (ret == 0)
            {
                fprintf(stderr, "encode image %ls failed\n", outputpngpath);
                return -1;
            }
#else
            stbi_image_free(rgbdata);

            int ret = stbi_write_png(outputpngpath, outrgb.w, outrgb.h, 3, outrgb.data, 0);
            if (ret == 0)
            {
                fprintf(stderr, "encode image %s failed\n", outputpngpath);
                return -1;
            }
#endif
        }

        // cleanup preprocess and postprocess pipeline
        {
            delete waifu2x_preproc;
            delete waifu2x_postproc;
        }
    }

    delete vkdev;

    ncnn::destroy_gpu_instance();

    return 0;
}
