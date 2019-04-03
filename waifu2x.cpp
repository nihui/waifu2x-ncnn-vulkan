// waifu2x implemented with ncnn library

#include <stdio.h>

// image decoder and encoder
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_PIC
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ncnn
#include "layer_type.h"
#include "net.h"
#include "gpu.h"

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s [image] [outputpng] [noise=-1/0/1/2/3] [scale=1/2]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];
    const char* outputpngpath = argv[2];
    int noise = atoi(argv[3]);
    int scale = atoi(argv[4]);

    if (noise < -1 || noise > 3 || scale < 1 || scale > 2)
    {
        fprintf(stderr, "invalid noise or scale argument\n");
        return -1;
    }

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

    ncnn::create_gpu_instance();

    ncnn::VulkanDevice* vkdev = new ncnn::VulkanDevice;

    {
        ncnn::Net waifu2x;

        waifu2x.use_vulkan_compute = true;
        waifu2x.set_vulkan_device(vkdev);

        waifu2x.load_param(parampath);
        waifu2x.load_model(modelpath);

        // preprocess and postprocess operator
        ncnn::Layer* pre_padding = 0;
        ncnn::Layer* normalize = 0;
        ncnn::Layer* denormalize = 0;
        ncnn::Layer* cast_float32_to_float16 = 0;
        ncnn::Layer* cast_float16_to_float32 = 0;

        ncnn::Option opt = ncnn::get_default_option();
        opt.blob_vkallocator = vkdev->allocator();
        opt.workspace_vkallocator = vkdev->allocator();
        opt.staging_vkallocator = vkdev->staging_allocator();

        // initialize preprocess and postprocess operator
        {
            {
                pre_padding = ncnn::create_layer(ncnn::LayerType::Padding);
                pre_padding->vkdev = vkdev;

                ncnn::ParamDict pd;
                pd.set(0, prepadding);
                pd.set(1, prepadding);
                pd.set(2, prepadding);
                pd.set(3, prepadding);
                pd.set(4, ncnn::BORDER_REPLICATE);
                pd.set(5, 0.f);
                pd.use_vulkan_compute = 1;

                pre_padding->load_param(pd);

                pre_padding->create_pipeline();
            }

            {
                normalize = ncnn::create_layer(ncnn::LayerType::BinaryOp);
                normalize->vkdev = vkdev;

                ncnn::ParamDict pd;
                pd.set(0, 2);
                pd.set(1, 1);
                pd.set(2, 1/255.f);
                pd.use_vulkan_compute = 1;

                normalize->load_param(pd);

                normalize->create_pipeline();
            }

            {
                denormalize = ncnn::create_layer(ncnn::LayerType::BinaryOp);
                denormalize->vkdev = vkdev;

                ncnn::ParamDict pd;
                pd.set(0, 2);
                pd.set(1, 1);
                pd.set(2, 255.f);
                pd.use_vulkan_compute = 1;

                denormalize->load_param(pd);

                denormalize->create_pipeline();
            }

            if (vkdev->info.support_fp16_storage)
            {
                {
                    cast_float32_to_float16 = ncnn::create_layer(ncnn::LayerType::Cast);
                    cast_float32_to_float16->vkdev = vkdev;

                    ncnn::ParamDict pd;
                    pd.set(0, 1);
                    pd.set(1, 2);
                    pd.use_vulkan_compute = 1;

                    cast_float32_to_float16->load_param(pd);

                    cast_float32_to_float16->create_pipeline();
                }

                {
                    cast_float16_to_float32 = ncnn::create_layer(ncnn::LayerType::Cast);
                    cast_float16_to_float32->vkdev = vkdev;

                    ncnn::ParamDict pd;
                    pd.set(0, 2);
                    pd.set(1, 1);
                    pd.use_vulkan_compute = 1;

                    cast_float16_to_float32->load_param(pd);

                    cast_float16_to_float32->create_pipeline();
                }
            }
        }

        // main routine
        {
            int w, h, c;
            unsigned char* rgbdata = stbi_load(imagepath, &w, &h, &c, 3);
            if (!rgbdata)
            {
                fprintf(stderr, "decode image %s failed\n", imagepath);
                return -1;
            }

            ncnn::Mat in = ncnn::Mat::from_pixels(rgbdata, ncnn::Mat::PIXEL_RGB, w, h);

            stbi_image_free(rgbdata);

            ncnn::VkCompute cmd(vkdev);

            // upload
            ncnn::VkMat in_gpu;
            {
                in_gpu.create_like(in, opt.blob_vkallocator, opt.staging_vkallocator);

                in_gpu.prepare_staging_buffer();
                in_gpu.upload(in);

                cmd.record_upload(in_gpu);
            }

            // cast to fp16
            if (vkdev->info.support_fp16_storage)
            {
                ncnn::VkMat in_gpu_fp16;
                cast_float32_to_float16->forward(in_gpu, in_gpu_fp16, cmd, opt);
                in_gpu = in_gpu_fp16;
            }

            // prepadding
            {
                ncnn::VkMat in_gpu_padded;
                pre_padding->forward(in_gpu, in_gpu_padded, cmd, opt);
                in_gpu = in_gpu_padded;
            }

            // normalize
            {
                ncnn::VkMat in_gpu_normed;
                normalize->forward(in_gpu, in_gpu_normed, cmd, opt);
                in_gpu = in_gpu_normed;
            }

            ncnn::Extractor ex = waifu2x.create_extractor();

            ex.input("Input1", in_gpu);

            ncnn::VkMat out_gpu;
            ex.extract("Eltwise4", out_gpu, cmd);

            // denormalize
            {
                ncnn::VkMat out_gpu_denormed;
                denormalize->forward(out_gpu, out_gpu_denormed, cmd, opt);
                out_gpu = out_gpu_denormed;
            }

            // cast to fp32
            if (vkdev->info.support_fp16_storage)
            {
                ncnn::VkMat out_gpu_fp32;
                cast_float16_to_float32->forward(out_gpu, out_gpu_fp32, cmd, opt);
                out_gpu = out_gpu_fp32;
            }

            // download
            {
                out_gpu.prepare_staging_buffer();
                cmd.record_download(out_gpu);
            }

            cmd.submit();

            cmd.wait();

            ncnn::Mat out;
            out.create_like(out_gpu, opt.blob_allocator);
            out_gpu.download(out);

            ncnn::Mat outrgb(out.w, out.h, 1, 3u, 3);
            out.to_pixels((unsigned char*)outrgb.data, ncnn::Mat::PIXEL_RGB);

            int ret = stbi_write_png(outputpngpath, outrgb.w, outrgb.h, 3, outrgb.data, 0);
            if (ret == 0)
            {
                fprintf(stderr, "encode image %s failed\n", outputpngpath);
                return -1;
            }
        }

        // cleanup preprocess and postprocess operator
        {
            if (vkdev->info.support_fp16_storage)
            {
                cast_float32_to_float16->destroy_pipeline();
                delete cast_float32_to_float16;

                cast_float16_to_float32->destroy_pipeline();
                delete cast_float16_to_float32;
            }

            pre_padding->destroy_pipeline();
            delete pre_padding;

            normalize->destroy_pipeline();
            delete normalize;

            denormalize->destroy_pipeline();
            delete denormalize;
        }
    }

    delete vkdev;

    ncnn::destroy_gpu_instance();

    return 0;
}
