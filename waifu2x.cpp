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

#if WIN32
    CoInitialize(0);
#endif
 
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
        ncnn::Layer* post_crop = 0;
        ncnn::Layer* normalize = 0;
        ncnn::Layer* denormalize = 0;
        ncnn::Layer* cast_float32_to_float16 = 0;
        ncnn::Layer* cast_float16_to_float32 = 0;
        ncnn::Layer* crop_tile = 0;
        ncnn::Layer* merge_tile_x = 0;
        ncnn::Layer* merge_tile_y = 0;

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
                pd.set(0, -233);
                pd.set(1, -233);
                pd.set(2, -233);
                pd.set(3, -233);
                pd.set(4, ncnn::BORDER_REPLICATE);
                pd.set(5, 0.f);
                pd.use_vulkan_compute = 1;

                pre_padding->load_param(pd);

                pre_padding->create_pipeline();
            }

            {
                post_crop = ncnn::create_layer(ncnn::LayerType::Crop);
                post_crop->vkdev = vkdev;

                ncnn::ParamDict pd;
                pd.set(0, 0);
                pd.set(1, 0);
                pd.set(2, 0);
                pd.use_vulkan_compute = 1;

                post_crop->load_param(pd);

                post_crop->create_pipeline();
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

            {
                crop_tile = ncnn::create_layer(ncnn::LayerType::Crop);
                crop_tile->vkdev = vkdev;

                ncnn::ParamDict pd;
                pd.set(0, -233);
                pd.set(1, -233);
                pd.set(2, -233);
                pd.set(3, 0);
                pd.set(4, 0);
                pd.set(5, 0);
                pd.use_vulkan_compute = 1;

                crop_tile->load_param(pd);

                crop_tile->create_pipeline();
            }

            {
                merge_tile_x = ncnn::create_layer(ncnn::LayerType::Concat);
                merge_tile_x->vkdev = vkdev;

                ncnn::ParamDict pd;
                pd.set(0, 2);
                pd.use_vulkan_compute = 1;

                merge_tile_x->load_param(pd);

                merge_tile_x->create_pipeline();
            }

            {
                merge_tile_y = ncnn::create_layer(ncnn::LayerType::Concat);
                merge_tile_y->vkdev = vkdev;

                ncnn::ParamDict pd;
                pd.set(0, 1);
                pd.use_vulkan_compute = 1;

                merge_tile_y->load_param(pd);

                merge_tile_y->create_pipeline();
            }
        }

        // main routine
        {
#if WIN32
            int w, h, c;
            unsigned char* bgrdata = wic_decode_image(imagepath, &w, &h, &c);
            if (!bgrdata)
            {
                fprintf(stderr, "decode image %s failed\n", imagepath);
                return -1;
            }

            ncnn::Mat in = ncnn::Mat::from_pixels(bgrdata, ncnn::Mat::PIXEL_BGR2RGB, w, h);

            free(bgrdata);
#else // WIN32
            int w, h, c;
            unsigned char* rgbdata = stbi_load(imagepath, &w, &h, &c, 3);
            if (!rgbdata)
            {
                fprintf(stderr, "decode image %s failed\n", imagepath);
                return -1;
            }

            ncnn::Mat in = ncnn::Mat::from_pixels(rgbdata, ncnn::Mat::PIXEL_RGB, w, h);

            stbi_image_free(rgbdata);
#endif // WIN32

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

            // normalize
            {
                ncnn::VkMat in_gpu_normed;
                normalize->forward(in_gpu, in_gpu_normed, cmd, opt);
                in_gpu = in_gpu_normed;
            }

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
            {
                ncnn::VkMat pre_padding_params(4, (size_t)4u, 1, opt.blob_vkallocator, opt.staging_vkallocator);
                pre_padding_params.prepare_staging_buffer();
                int* padding_param = pre_padding_params.mapped();

                padding_param[0] = prepadding;
                padding_param[1] = prepadding_bottom;
                padding_param[2] = prepadding;
                padding_param[3] = prepadding_right;

                std::vector<ncnn::VkMat> pre_padding_inputs(2);
                pre_padding_inputs[0] = in_gpu;
                pre_padding_inputs[1] = pre_padding_params;

                std::vector<ncnn::VkMat> pre_padding_outputs(1);
                pre_padding->forward(pre_padding_inputs, pre_padding_outputs, cmd, opt);
                in_gpu = pre_padding_outputs[0];
            }

            ncnn::VkMat out_gpu;

            // each tile 400x400
            {
                const int TILE_SIZE_X = 400;
                const int TILE_SIZE_Y = 400;

                ncnn::VkMat crop_tile_params(6, (size_t)4u, 1, opt.blob_vkallocator, opt.staging_vkallocator);
                crop_tile_params.prepare_staging_buffer();
                int* crop_param = crop_tile_params.mapped();

                int xtiles = (w + TILE_SIZE_X-1) / TILE_SIZE_X;
                int ytiles = (h + TILE_SIZE_Y-1) / TILE_SIZE_Y;

                std::vector<ncnn::VkMat> out_tile_y_gpus(ytiles);
                for (int yi = 0; yi < ytiles; yi++)
                {
                    std::vector<ncnn::VkMat> out_tile_x_gpus(xtiles);
                    for (int xi = 0; xi < xtiles; xi++)
                    {
                        // crop tile
                        int tile_x0 = xi * TILE_SIZE_X;
                        int tile_x1 = std::min((xi+1) * TILE_SIZE_X, w) + prepadding + prepadding_right;
                        int tile_y0 = yi * TILE_SIZE_Y;
                        int tile_y1 = std::min((yi+1) * TILE_SIZE_Y, h) + prepadding + prepadding_bottom;

                        crop_param[0] = tile_x0;
                        crop_param[1] = tile_y0;
                        crop_param[2] = 0;
                        crop_param[3] = tile_x1 - tile_x0;
                        crop_param[4] = tile_y1 - tile_y0;
                        crop_param[5] = 3;

                        std::vector<ncnn::VkMat> crop_tile_inputs(2);
                        crop_tile_inputs[0] = in_gpu;
                        crop_tile_inputs[1] = crop_tile_params;

                        std::vector<ncnn::VkMat> crop_tile_outputs(1);
                        crop_tile->forward(crop_tile_inputs, crop_tile_outputs, cmd, opt);

                        ncnn::VkMat in_tile_gpu = crop_tile_outputs[0];

                        // waifu2x
                        {
                            ncnn::Extractor ex = waifu2x.create_extractor();
                            ex.input("Input1", in_tile_gpu);

                            ex.extract("Eltwise4", out_tile_x_gpus[ xi ], cmd);
                        }
                    }

                    // merge tiles x
                    std::vector<ncnn::VkMat> merge_tile_x_outputs(1);
                    merge_tile_x->forward(out_tile_x_gpus, merge_tile_x_outputs, cmd, opt);
                    out_tile_y_gpus[ yi ] = merge_tile_x_outputs[0];
                }

                // merge tiles y
                std::vector<ncnn::VkMat> merge_tile_y_outputs(1);
                merge_tile_y->forward(out_tile_y_gpus, merge_tile_y_outputs, cmd, opt);
                out_gpu = merge_tile_y_outputs[0];
            }

            // postcrop
            {
                ncnn::VkMat crop_reference_blob;
                crop_reference_blob.dims = 2;
                crop_reference_blob.w = w * scale;
                crop_reference_blob.h = h * scale;

                std::vector<ncnn::VkMat> post_crop_inputs(2);
                post_crop_inputs[0] = out_gpu;
                post_crop_inputs[1] = crop_reference_blob;

                std::vector<ncnn::VkMat> post_crop_outputs(1);
                post_crop->forward(post_crop_inputs, post_crop_outputs, cmd, opt);
                out_gpu = post_crop_outputs[0];
            }

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

#if WIN32
            ncnn::Mat outbgr(out.w, out.h, (size_t)3u, 3);
            out.to_pixels((unsigned char*)outbgr.data, ncnn::Mat::PIXEL_RGB2BGR);

            int ret = wic_encode_image(outputpngpath, outbgr.w, outbgr.h, 3, outbgr.data);
#else
            ncnn::Mat outrgb(out.w, out.h, (size_t)3u, 3);
            out.to_pixels((unsigned char*)outrgb.data, ncnn::Mat::PIXEL_RGB);

            int ret = stbi_write_png(outputpngpath, outrgb.w, outrgb.h, 3, outrgb.data, 0);
#endif
            if (ret == 0)
            {
                fprintf(stderr, "encode image %s failed\n", outputpngpath);
                return -1;
            }
        }

        // cleanup preprocess and postprocess operator
        {
            merge_tile_x->destroy_pipeline();
            delete merge_tile_x;

            merge_tile_y->destroy_pipeline();
            delete merge_tile_y;

            crop_tile->destroy_pipeline();
            delete crop_tile;

            if (vkdev->info.support_fp16_storage)
            {
                cast_float32_to_float16->destroy_pipeline();
                delete cast_float32_to_float16;

                cast_float16_to_float32->destroy_pipeline();
                delete cast_float16_to_float32;
            }

            pre_padding->destroy_pipeline();
            delete pre_padding;

            post_crop->destroy_pipeline();
            delete post_crop;

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
