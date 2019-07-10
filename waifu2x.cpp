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

using namespace std;

#if WIN32
#include <wchar.h>
static wchar_t* optarg = NULL;
static int optind = 1;
static wchar_t getopt(int argc, wchar_t* const argv[], const wchar_t* optstring)
{
	if (optind >= argc || argv[optind][0] != L'-')
		return -1;

	wchar_t opt = argv[optind][1];
	const wchar_t* p = wcschr(optstring, opt);
	if (p == NULL)
		return L'?';

	optarg = NULL;

	if (p[1] == L':')
	{
		optind++;
		if (optind >= argc)
			return L'?';

		optarg = argv[optind];
	}

	optind++;

	return opt;
}
#else // WIN32
#include <unistd.h> // getopt()
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
#if !defined(NO_INT8_SUPPORT)
static const uint32_t waifu2x_preproc_int8s_spv_data[] = {
	#include "waifu2x_preproc_int8s.spv.hex.h"
};
#endif
static const uint32_t waifu2x_postproc_spv_data[] = {
	#include "waifu2x_postproc.spv.hex.h"
};
static const uint32_t waifu2x_postproc_fp16s_spv_data[] = {
	#include "waifu2x_postproc_fp16s.spv.hex.h"
};
#if !defined(NO_INT8_SUPPORT)
static const uint32_t waifu2x_postproc_int8s_spv_data[] = {
	#include "waifu2x_postproc_int8s.spv.hex.h"
};
#endif

class waifu2x {
private:
	ncnn::Net net;
	ncnn::VulkanDevice* vkdev;
	ncnn::Pipeline* preproc;
	ncnn::Pipeline* postproc;
	int prepadding = 0;
	int scale = 0;
	int tilesize = 0;
#if WIN32
	const wchar_t* model = L"models-cunet";
	wchar_t parampath[256];
	wchar_t modelpath[256];
#else // WIN32
	const char* model = "models-cunet";
	char parampath[256];
	char modelpath[256];
#endif

public:
#ifdef WIN32
	waifu2x(int gpuid = 0, int noise = 0, int scale = 2, int tilesize = 400, const wchar_t* model = 0) {
#else
	waifu2x(int gpuid = 0, int noise = 0, int scale = 2, int tilesize = 400, const char* model = 0) {
#endif
		if (model) {
			this->model = model;
		}
		if (noise < -1 || noise > 3 || scale < 1 || scale > 2)
		{
			fprintf(stderr, "invalid noise or scale argument\n");
			throw - 1;
		}

		if (tilesize < 32)
		{
			fprintf(stderr, "invalid tilesize argument\n");
			throw - 1;
		}

		this->scale = scale;
		this->tilesize = tilesize;

		ncnn::create_gpu_instance();

		int gpu_count = ncnn::get_gpu_count();
		if (gpuid < 0 || gpuid >= gpu_count)
		{
			fprintf(stderr, "invalid gpu device");
			ncnn::destroy_gpu_instance();
			throw - 1;
		}
		this->vkdev = ncnn::get_gpu_device(gpuid);

		ncnn::VkAllocator* blob_vkallocator = this->vkdev->acquire_blob_allocator();
		ncnn::VkAllocator* staging_vkallocator = this->vkdev->acquire_staging_allocator();

		ncnn::Option opt;
		opt.use_vulkan_compute = true;
		opt.blob_vkallocator = blob_vkallocator;
		opt.workspace_vkallocator = blob_vkallocator;
		opt.staging_vkallocator = staging_vkallocator;
		opt.use_fp16_packed = true;
		opt.use_fp16_storage = true;
		opt.use_fp16_arithmetic = false;
#if !defined(NO_INT8_SUPPORT)
		opt.use_int8_storage = true;
#else
		opt.use_int8_storage = false;
#endif
		opt.use_int8_arithmetic = false;

		this->net.opt = opt;
		this->net.set_vulkan_device(this->vkdev);
		this->init_proc();
		this->set_params(noise, scale);
		this->load_models();
	}
	~waifu2x()
	{
		// cleanup preprocess and postprocess pipeline
		delete this->preproc;
		delete this->postproc;

		this->vkdev->reclaim_blob_allocator(this->net.opt.blob_vkallocator);
		this->vkdev->reclaim_staging_allocator(this->net.opt.staging_vkallocator);

		ncnn::destroy_gpu_instance();
	}
private:
	void set_params(int noise, int scale) {
#if WIN32
		if (wcsstr(this->model, L"models-cunet"))
#else
		if (strstr(this->model, "models-cunet"))
#endif
		{
			if (noise == -1)
			{
				this->prepadding = 18;
			}
			else if (scale == 1)
			{
				this->prepadding = 28;
			}
			else if (scale == 2)
			{
				this->prepadding = 18;
			}
		}
#if WIN32
		else if (wcsstr(this->model, L"models-upconv_7_anime_style_art_rgb"))
#else
		else if (strstr(this->model, "models-upconv_7_anime_style_art_rgb"))
#endif
		{
			this->prepadding = 7;
		}
		else
		{
			fprintf(stderr, "unknown model dir type");
			throw - 1;
		}

#if WIN32
		if (noise == -1)
		{
			swprintf(this->parampath, 256, L"%s/scale2.0x_model.param", this->model);
			swprintf(this->modelpath, 256, L"%s/scale2.0x_model.bin", this->model);
		}
		else if (scale == 1)
		{
			swprintf(this->parampath, 256, L"%s/noise%d_model.param", this->model, noise);
			swprintf(this->modelpath, 256, L"%s/noise%d_model.bin", this->model, noise);
		}
		else if (scale == 2)
		{
			swprintf(this->parampath, 256, L"%s/noise%d_scale2.0x_model.param", this->model, noise);
			swprintf(this->modelpath, 256, L"%s/noise%d_scale2.0x_model.bin", this->model, noise);
		}
#else
		if (noise == -1)
		{
			sprintf(this->parampath, "%s/scale2.0x_model.param", this->model);
			sprintf(this->modelpath, "%s/scale2.0x_model.bin", this->model);
		}
		else if (scale == 1)
		{
			sprintf(this->parampath, "%s/noise%d_model.param", this->model, noise);
			sprintf(this->modelpath, "%s/noise%d_model.bin", this->model, noise);
		}
		else if (scale == 2)
		{
			sprintf(this->parampath, "%s/noise%d_scale2.0x_model.param", this->model, noise);
			sprintf(this->modelpath, "%s/noise%d_scale2.0x_model.bin", this->model, noise);
		}
#endif
	}
	void load_models() {
#if WIN32
		{
			FILE* fp = _wfopen(this->parampath, L"rb");
			if (!fp)
			{
				fwprintf(stderr, L"_wfopen %s failed\n", this->parampath);
				throw - 1;
			}

			this->net.load_param(fp);

			fclose(fp);
		}
		{
			FILE* fp = _wfopen(this->modelpath, L"rb");
			if (!fp)
			{
				fwprintf(stderr, L"_wfopen %s failed\n", this->modelpath);
				throw - 1;
			}

			this->net.load_model(fp);

			fclose(fp);
		}
#else
		this->net..load_param(parampath);
		this->net.load_model(modelpath);
#endif
	}
	void init_proc() {
		// initialize preprocess and postprocess pipeline

		vector<ncnn::vk_specialization_type> specializations(1);
#if WIN32
		specializations[0].i = 1;
#else
		specializations[0].i = 0;
#endif

		this->preproc = new ncnn::Pipeline(this->vkdev);
		this->preproc->set_optimal_local_size_xyz(32, 32, 3);
#if !defined(NO_INT8_SUPPORT)
		if (this->net.opt.use_fp16_storage && this->net.opt.use_int8_storage)
			this->preproc->create(waifu2x_preproc_int8s_spv_data, sizeof(waifu2x_preproc_int8s_spv_data), "waifu2x_preproc_int8s", specializations, 2, 9);
		else
#endif
			if (this->net.opt.use_fp16_storage)
				this->preproc->create(waifu2x_preproc_fp16s_spv_data, sizeof(waifu2x_preproc_fp16s_spv_data), "waifu2x_preproc_fp16s", specializations, 2, 9);
			else
				this->preproc->create(waifu2x_preproc_spv_data, sizeof(waifu2x_preproc_spv_data), "waifu2x_preproc", specializations, 2, 9);

		this->postproc = new ncnn::Pipeline(this->vkdev);
		this->postproc->set_optimal_local_size_xyz(32, 32, 3);
#if !defined(NO_INT8_SUPPORT)
		if (this->net.opt.use_fp16_storage && this->net.opt.use_int8_storage)
			this->postproc->create(waifu2x_postproc_int8s_spv_data, sizeof(waifu2x_postproc_int8s_spv_data), "waifu2x_postproc_int8s", specializations, 2, 8);
		else
#endif
			if (this->net.opt.use_fp16_storage)
				this->postproc->create(waifu2x_postproc_fp16s_spv_data, sizeof(waifu2x_postproc_fp16s_spv_data), "waifu2x_postproc_fp16s", specializations, 2, 8);
			else
				this->postproc->create(waifu2x_postproc_spv_data, sizeof(waifu2x_postproc_spv_data), "waifu2x_postproc", specializations, 2, 8);
	}
public:
#ifdef WIN32
	void proc_image(const wchar_t* imagepath = 0, const wchar_t* outputpngpath = 0) {
#else
	void proc_image(const char* imagepath = 0, const char* outputpngpath = 0) {
#endif
		const int TILE_SIZE_X = this->tilesize;
		const int TILE_SIZE_Y = this->tilesize;
#if WIN32
		int w, h, c;
		unsigned char* bgrdata = wic_decode_image(imagepath, &w, &h, &c);
		if (!bgrdata)
		{
			fprintf(stderr, "decode image %ls failed\n", imagepath);
			throw - 1;
		}

		ncnn::Mat outbgr(w * this->scale, h * this->scale, (size_t)3u, 3);
#else // WIN32
		int w, h, c;
		unsigned char* rgbdata = stbi_load(imagepath, &w, &h, &c, 3);
		if (!rgbdata)
		{
			fprintf(stderr, "decode image %s failed\n", imagepath);
			throw - 1;
		}

		ncnn::Mat outrgb(w * scale, h * scale, (size_t)3u, 3);
#endif // WIN32

		// prepadding
		int prepadding_bottom = this->prepadding;
		int prepadding_right = this->prepadding;
		if (this->scale == 1)
		{
			prepadding_bottom += (h + 3) / 4 * 4 - h;
			prepadding_right += (w + 3) / 4 * 4 - w;
		}
		if (this->scale == 2)
		{
			prepadding_bottom += (h + 1) / 2 * 2 - h;
			prepadding_right += (w + 1) / 2 * 2 - w;
		}

		// each tile 400x400
		int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
		int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

		//#pragma omp parallel for num_threads(2)
		for (int yi = 0; yi < ytiles; yi++)
		{
			int in_tile_y0 = max(yi * TILE_SIZE_Y - this->prepadding, 0);
			int in_tile_y1 = min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);

			ncnn::Mat in;
			if (this->net.opt.use_fp16_storage && this->net.opt.use_int8_storage)
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

			ncnn::VkCompute cmd(this->vkdev);

			// upload
			ncnn::VkMat in_gpu;
			{
				in_gpu.create_like(in, this->net.opt.blob_vkallocator, this->net.opt.staging_vkallocator);

				in_gpu.prepare_staging_buffer();
				in_gpu.upload(in);

				cmd.record_upload(in_gpu);

				if (xtiles > 1)
				{
					cmd.submit_and_wait();
					cmd.reset();
				}
			}

			int out_tile_y0 = max(yi * TILE_SIZE_Y, 0);
			int out_tile_y1 = min((yi + 1) * TILE_SIZE_Y, h);

			ncnn::VkMat out_gpu;
			if (this->net.opt.use_fp16_storage && this->net.opt.use_int8_storage)
			{
				out_gpu.create(w * this->scale, (out_tile_y1 - out_tile_y0) * this->scale, (size_t)3u, 1, this->net.opt.blob_vkallocator, this->net.opt.staging_vkallocator);
			}
			else
			{
				out_gpu.create(w * this->scale, (out_tile_y1 - out_tile_y0) * this->scale, 3, (size_t)4u, 1, this->net.opt.blob_vkallocator, this->net.opt.staging_vkallocator);
			}

			for (int xi = 0; xi < xtiles; xi++)
			{
				// preproc
				ncnn::VkMat in_tile_gpu;
				{
					// crop tile
					int tile_x0 = xi * TILE_SIZE_X;
					int tile_x1 = min((xi + 1) * TILE_SIZE_X, w) + prepadding + prepadding_right;
					int tile_y0 = yi * TILE_SIZE_Y;
					int tile_y1 = min((yi + 1) * TILE_SIZE_Y, h) + prepadding + prepadding_bottom;

					in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, (size_t)4u, 1, this->net.opt.blob_vkallocator, this->net.opt.staging_vkallocator);

					vector<ncnn::VkMat> bindings(2);
					bindings[0] = in_gpu;
					bindings[1] = in_tile_gpu;

					vector<ncnn::vk_constant_type> constants(9);
					constants[0].i = in_gpu.w;
					constants[1].i = in_gpu.h;
					constants[2].i = in_gpu.cstep;
					constants[3].i = in_tile_gpu.w;
					constants[4].i = in_tile_gpu.h;
					constants[5].i = in_tile_gpu.cstep;
					constants[6].i = max(this->prepadding - yi * TILE_SIZE_Y, 0);
					constants[7].i = this->prepadding;
					constants[8].i = xi * TILE_SIZE_X;

					cmd.record_pipeline(this->preproc, bindings, constants, in_tile_gpu);
				}

				// waifu2x
				ncnn::VkMat out_tile_gpu;
				{
					ncnn::Extractor ex = this->net.create_extractor();
					ex.input("Input1", in_tile_gpu);

					ex.extract("Eltwise4", out_tile_gpu, cmd);
				}

				// postproc
				{
					vector<ncnn::VkMat> bindings(2);
					bindings[0] = out_tile_gpu;
					bindings[1] = out_gpu;

					vector<ncnn::vk_constant_type> constants(8);
					constants[0].i = out_tile_gpu.w;
					constants[1].i = out_tile_gpu.h;
					constants[2].i = out_tile_gpu.cstep;
					constants[3].i = out_gpu.w;
					constants[4].i = out_gpu.h;
					constants[5].i = out_gpu.cstep;
					constants[6].i = xi * TILE_SIZE_X * this->scale;
					constants[7].i = out_gpu.w - xi * TILE_SIZE_X * this->scale;

					ncnn::VkMat dispatcher;
					dispatcher.w = out_gpu.w - xi * TILE_SIZE_X * this->scale;
					dispatcher.h = out_gpu.h;
					dispatcher.c = 3;

					cmd.record_pipeline(this->postproc, bindings, constants, dispatcher);
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

			if (this->net.opt.use_fp16_storage && this->net.opt.use_int8_storage)
			{
#if WIN32
				ncnn::Mat out(out_gpu.w, out_gpu.h, (unsigned char*)outbgr.data + yi * this->scale * TILE_SIZE_Y * w * this->scale * 3, (size_t)3u, 1);
#else
				ncnn::Mat out(out_gpu.w, out_gpu.h, (unsigned char*)outrgb.data + yi * scale * TILE_SIZE_Y * w * scale * 3, (size_t)3u, 1);
#endif

				out_gpu.download(out);
			}
			else
			{
				ncnn::Mat out;
				out.create_like(out_gpu, this->net.opt.blob_allocator);
				out_gpu.download(out);

#if WIN32
				out.to_pixels((unsigned char*)outbgr.data + yi * this->scale * TILE_SIZE_Y * w * this->scale * 3, ncnn::Mat::PIXEL_RGB2BGR);
#else
				out.to_pixels((unsigned char*)outrgb.data + yi * this->scale * TILE_SIZE_Y * w * this->scale * 3, ncnn::Mat::PIXEL_RGB);
#endif
			}
		}

#if WIN32
		free(bgrdata);

		int ret = wic_encode_image(outputpngpath, outbgr.w, outbgr.h, 3, outbgr.data);
		if (ret == 0)
		{
			fprintf(stderr, "encode image %ls failed\n", outputpngpath);
			throw - 1;
		}
#else
		stbi_image_free(rgbdata);

		int ret = stbi_write_png(outputpngpath, outrgb.w, outrgb.h, 3, outrgb.data, 0);
		if (ret == 0)
		{
			fprintf(stderr, "encode image %s failed\n", outputpngpath);
			throw - 1;
		}
#endif
	}
};

static void print_usage()
{
	fprintf(stderr, "Usage: waifu2x-ncnn-vulkan -i infile -o outfile [options]...\n\n");
	fprintf(stderr, "  -h               show this help\n");
	fprintf(stderr, "  -i input-image   input image path (jpg/png)\n");
	fprintf(stderr, "  -o output-image  output image path (png)\n");
	fprintf(stderr, "  -n noise-level   denoise level (-1/0/1/2/3, default=0)\n");
	fprintf(stderr, "  -s scale         upscale ratio (1/2, default=2)\n");
	fprintf(stderr, "  -t tile-size     tile size (>=32, default=400)\n");
	fprintf(stderr, "  -m model-path    waifu2x model path (default=models-cunet)\n");
	fprintf(stderr, "  -g gpu-id        gpu device to use (default=0)\n");
}

#if WIN32
int wmain(int argc, wchar_t** argv)
#else
int main(int argc, char** argv)
#endif
{
#if WIN32
	const wchar_t* imagepath = 0;
	const wchar_t* outputpngpath = 0;
	int noise = 0;
	int scale = 2;
	int tilesize = 400;
	const wchar_t* model = 0;
	int gpuid = 0;

	wchar_t opt;
	while ((opt = getopt(argc, argv, L"i:o:n:s:t:m:g:h")) != (wchar_t)-1)
	{
		switch (opt)
		{
		case L'i':
			imagepath = optarg;
			break;
		case L'o':
			outputpngpath = optarg;
			break;
		case L'n':
			noise = _wtoi(optarg);
			break;
		case L's':
			scale = _wtoi(optarg);
			break;
		case L't':
			tilesize = _wtoi(optarg);
			break;
		case L'm':
			model = optarg;
			break;
		case L'g':
			gpuid = _wtoi(optarg);
			break;
		case L'h':
		default:
			print_usage();
			return -1;
		}
	}
#else // WIN32
	const char* imagepath = 0;
	const char* outputpngpath = 0;
	int noise = 0;
	int scale = 2;
	int tilesize = 400;
	const char* model = "models-cunet";
	int gpuid = 0;

	int opt;
	while ((opt = getopt(argc, argv, "i:o:n:s:t:m:g:h")) != -1)
	{
		switch (opt)
		{
		case 'i':
			imagepath = optarg;
			break;
		case 'o':
			outputpngpath = optarg;
			break;
		case 'n':
			noise = atoi(optarg);
			break;
		case 's':
			scale = atoi(optarg);
			break;
		case 't':
			tilesize = atoi(optarg);
			break;
		case 'm':
			model = optarg;
			break;
		case 'g':
			gpuid = atoi(optarg);
			break;
		case 'h':
		default:
			print_usage();
			return -1;
		}
	}
#endif // WIN32

	if (!imagepath || !outputpngpath)
	{
		print_usage();
		return -1;
	}

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

#if WIN32
	CoInitialize(0);
#endif

	waifu2x* processer = new waifu2x(gpuid, noise, scale, tilesize);
	fwprintf(stderr, outputpngpath);
	processer->proc_image(imagepath, outputpngpath);
	delete processer;

	return 0;
}
