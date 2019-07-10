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


class waifu2x_config {
public:
	int prepadding = 0;
	int noise = 0;
	int scale = 2;
	int tilesize = 128;
private:
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
	waifu2x_config(int noise = 0, int scale = 2, int tilesize = 400, const wchar_t* model = 0)
		:parampath(L""), modelpath(L"") {
#else
	waifu2x_config(int noise = 0, int scale = 2, int tilesize = 400, const char* model = 0)
		:parampath(""), modelpath("") {
#endif
		if (noise >= 0) {
			this->noise = noise;
		}
		if (scale >= 1) {
			this->noise = scale;
		}
		if (tilesize >= 32) {
			this->tilesize = tilesize;
		}
		if (model) {
			this->model = model;
		}
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
			return;
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
#ifdef WIN32
	FILE* read_param() {
		return read_file(this->parampath);
	}
	FILE* read_model() {
		return read_file(this->modelpath);
	}
private:
	FILE* read_file(wchar_t* path) {
		FILE* fp = _wfopen(path, L"rb");
		if (!fp)
		{
			fwprintf(stderr, L"_wfopen %s failed\n", path);
			return nullptr;
		}
		return fp;
	}
#else
	char* read_param() {
		return this->parampath;
	}
	char* read_model() {
		return this->modelpath;
	}
#endif
};

class waifu2x_image {
public:
	const waifu2x_config* config;
	int w, h, c;
	int prepadding_bottom, prepadding_right;
	int xtiles, ytiles;
	const int TILE_SIZE_X, TILE_SIZE_Y;
	unsigned char* data;
	ncnn::Mat buffer;
	waifu2x_image(waifu2x_config* config = new waifu2x_config())
		:config(config), TILE_SIZE_X(config->tilesize), TILE_SIZE_Y(config->tilesize),
		w(0), h(0), c(0), prepadding_bottom(0), prepadding_right(0), xtiles(0), ytiles(0), data(0)
	{}
	~waifu2x_image() {
#ifdef WIN32
		free(this->data);
#else
		stbi_image_free(this->data);
#endif
	}
#ifdef WIN32
	void encode(const wchar_t* output) {
#else
	void encode(const char* output) {
#endif
#if WIN32
		int ret = wic_encode_image(output, this->buffer.w, this->buffer.h, 3, this->buffer.data);
#else
		int ret = stbi_write_png(outputpngpath, outrgb.w, outrgb.h, 3, outrgb.data, 0);
#endif
		if (ret == 0)
		{
			fprintf(stderr, "encode image %ls failed\n", output);
			return;
		}
	}
#ifdef WIN32
	void decode(const wchar_t* input) {
#else
	void decode(const char* input) {
#endif
#if WIN32
		this->data = wic_decode_image(input, &this->w, &this->h, &this->c);
#else
		this->data = stbi_load(imagepath, &this->w, &this->h, &this->c, 3);
#endif 
		if (!this->data)
		{
			fprintf(stderr, "decode image %ls failed\n", input);
			return;
		}
		this->calc_config();
		this->buffer = ncnn::Mat(this->w * this->config->scale, this->h * this->config->scale, (size_t)3u, 3);
	}

private:
	void calc_config() {
		// prepadding
		int prepadding_bottom = this->config->prepadding;
		int prepadding_right = this->config->prepadding;
		if (this->config->scale == 1)
		{
			prepadding_bottom += (this->h + 3) / 4 * 4 - this->h;
			prepadding_right += (this->w + 3) / 4 * 4 - this->w;
		}
		if (this->config->scale == 2)
		{
			prepadding_bottom += (this->h + 1) / 2 * 2 - this->h;
			prepadding_right += (this->w + 1) / 2 * 2 - this->w;
		}
		this->prepadding_bottom = prepadding_bottom;
		this->prepadding_right = prepadding_right;
		this->xtiles = (this->w + this->TILE_SIZE_X - 1) / this->TILE_SIZE_X;
		this->ytiles = (this->h + this->TILE_SIZE_Y - 1) / this->TILE_SIZE_Y;
	}
};

class waifu2x {
private:
	ncnn::Net net;
	ncnn::VulkanDevice* vkdev;
	ncnn::Pipeline* preproc;
	ncnn::Pipeline* postproc;
	waifu2x_config config;

public:
	waifu2x(waifu2x_config config, int gpuid = 0) {
		this->config = config;
		ncnn::create_gpu_instance();

		int gpu_count = ncnn::get_gpu_count();
		if (gpuid < 0 || gpuid >= gpu_count)
		{
			fprintf(stderr, "invalid gpu device");
			ncnn::destroy_gpu_instance();
			return;
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
	void load_models() {
		this->net.load_param(this->config.read_param());
		this->net.load_model(this->config.read_model());
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
	void proc_image(waifu2x_image* image) {
		//#pragma omp parallel for num_threads(2)
		for (int yi = 0; yi < image->ytiles; yi++)
		{
			int in_tile_y0 = max(yi * image->TILE_SIZE_Y - image->config->prepadding, 0);
			int in_tile_y1 = min((yi + 1) * image->TILE_SIZE_Y + image->prepadding_bottom, image->h);

			ncnn::Mat in;
			if (this->net.opt.use_fp16_storage && this->net.opt.use_int8_storage)
			{
				in = ncnn::Mat(image->w, (in_tile_y1 - in_tile_y0), image->data + in_tile_y0 * image->w * 3, (size_t)3u, 1);
			}
			else
			{
#if WIN32
				in = ncnn::Mat::from_pixels(image->data + in_tile_y0 * image->w * 3, ncnn::Mat::PIXEL_BGR2RGB, image->w, (in_tile_y1 - in_tile_y0));
#else
				in = ncnn::Mat::from_pixels(image->data + in_tile_y0 * image->w * 3, ncnn::Mat::PIXEL_RGB, image->w, (in_tile_y1 - in_tile_y0));
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

				if (image->xtiles > 1)
				{
					cmd.submit_and_wait();
					cmd.reset();
				}
			}

			int out_tile_y0 = max(yi * image->TILE_SIZE_Y, 0);
			int out_tile_y1 = min((yi + 1) * image->TILE_SIZE_Y, image->h);

			ncnn::VkMat out_gpu;
			if (this->net.opt.use_fp16_storage && this->net.opt.use_int8_storage)
			{
				out_gpu.create(image->w * this->config.scale, (out_tile_y1 - out_tile_y0) * this->config.scale, (size_t)3u, 1, this->net.opt.blob_vkallocator, this->net.opt.staging_vkallocator);
			}
			else
			{
				out_gpu.create(image->w * this->config.scale, (out_tile_y1 - out_tile_y0) * this->config.scale, 3, (size_t)4u, 1, this->net.opt.blob_vkallocator, this->net.opt.staging_vkallocator);
			}

			for (int xi = 0; xi < image->xtiles; xi++)
			{
				// preproc
				ncnn::VkMat in_tile_gpu;
				{
					// crop tile
					int tile_x0 = xi * image->TILE_SIZE_X;
					int tile_x1 = min((xi + 1) * image->TILE_SIZE_X, image->w) + this->config.prepadding + image->prepadding_right;
					int tile_y0 = yi * image->TILE_SIZE_Y;
					int tile_y1 = min((yi + 1) * image->TILE_SIZE_Y, image->h) + this->config.prepadding + image->prepadding_bottom;

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
					constants[6].i = max(this->config.prepadding - yi * image->TILE_SIZE_Y, 0);
					constants[7].i = this->config.prepadding;
					constants[8].i = xi * image->TILE_SIZE_X;

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
					constants[6].i = xi * image->TILE_SIZE_X * this->config.scale;
					constants[7].i = out_gpu.w - xi * image->TILE_SIZE_X * this->config.scale;

					ncnn::VkMat dispatcher;
					dispatcher.w = out_gpu.w - xi * image->TILE_SIZE_X * this->config.scale;
					dispatcher.h = out_gpu.h;
					dispatcher.c = 3;

					cmd.record_pipeline(this->postproc, bindings, constants, dispatcher);
				}

				if (image->xtiles > 1)
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
				ncnn::Mat out(out_gpu.w, out_gpu.h, (unsigned char*)image->buffer.data + yi * this->config.scale * image->TILE_SIZE_Y * image->w * this->config.scale * 3, (size_t)3u, 1);
				out_gpu.download(out);
			}
			else
			{
				ncnn::Mat out;
				out.create_like(out_gpu, this->net.opt.blob_allocator);
				out_gpu.download(out);
#if WIN32
				out.to_pixels((unsigned char*)image->buffer.data + yi * this->config.scale * image->TILE_SIZE_Y * image->w * this->config.scale * 3, ncnn::Mat::PIXEL_RGB2BGR);
#else
				out.to_pixels((unsigned char*)image->buffer.data + yi * this->config.scale * image->TILE_SIZE_Y * image->w * this->config.scale * 3, ncnn::Mat::PIXEL_RGB);
#endif
			}
			}
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
	auto config = waifu2x_config(noise, scale, tilesize, model);
	auto image = new waifu2x_image(&config);
	auto processer = new waifu2x(config, gpuid);
	image->decode(imagepath);
	processer->proc_image(image);
	image->encode(outputpngpath);
	delete processer;

	return 0;
}
