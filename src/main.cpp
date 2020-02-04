// waifu2x implemented with ncnn library

#include <stdio.h>
#include <algorithm>
#include <queue>
#include <vector>
#include <locale>
  
#if _WIN32
// image decoder and encoder with wic
#include "wic_image.h"
#else // _WIN32
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
#endif // _WIN32

#if _WIN32
#include <wchar.h>
static wchar_t* optarg = NULL;
static int optind = 1;
static wchar_t getopt(int argc, wchar_t* const argv[], const wchar_t* optstring)
{
	setlocale( LC_ALL, "en_US.UTF-8" );
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
#else // _WIN32
#include <unistd.h> // getopt()
#endif // _WIN32

// ncnn
#include "cpu.h"
#include "gpu.h"
#include "platform.h"

#include "waifu2x.h"

#include "filesystem_utils.h"

static void print_usage()
{
    fprintf(stderr, "Usage: waifu2x-ncnn-vulkan -i infile -o outfile [options]...\n\n");
    fprintf(stderr, "  -h                   show this help\n");
    fprintf(stderr, "  -v                   verbose output\n");
    fprintf(stderr, "  -i input-path        input image path (jpg/png) or directory\n");
    fprintf(stderr, "  -o output-path       output image path (png) or directory\n");
    fprintf(stderr, "  -n noise-level       denoise level (-1/0/1/2/3, default=0)\n");
    fprintf(stderr, "  -s scale             upscale ratio (1/2, default=2)\n");
    fprintf(stderr, "  -t tile-size         tile size (>=32, default=400)\n");
    fprintf(stderr, "  -m model-path        waifu2x model path (default=models-cunet)\n");
    fprintf(stderr, "  -g gpu-id            gpu device to use (default=0)\n");
    fprintf(stderr, "  -j load:proc:save    thread count for load/proc/save (default=1:2:2)\n");
}

class Task
{
public:
    int id;

    path_t inpath;
    path_t outpath;

    ncnn::Mat inimage;
    ncnn::Mat outimage;
};

class TaskQueue
{
public:
    TaskQueue()
    {
    }

    void put(const Task& v)
    {
        lock.lock();

        while (tasks.size() >= 8) // FIXME hardcode queue length
        {
            condition.wait(lock);
        }

        tasks.push(v);

        lock.unlock();

        condition.signal();
    }

    void get(Task& v)
    {
        lock.lock();

        while (tasks.size() == 0)
        {
            condition.wait(lock);
        }

        v = tasks.front();
        tasks.pop();

        lock.unlock();

        condition.signal();
    }

private:
    ncnn::Mutex lock;
    ncnn::ConditionVariable condition;
    std::queue<Task> tasks;
};

TaskQueue toproc;
TaskQueue tosave;

class LoadThreadParams
{
public:
    int scale;
    int jobs_load;

    // session data
    std::vector<path_t> input_files;
    std::vector<path_t> output_files;
};

void* load(void* args)
{
    const LoadThreadParams* ltp = (const LoadThreadParams*)args;
    const int count = ltp->input_files.size();
    const int scale = ltp->scale;

    #pragma omp parallel for num_threads(ltp->jobs_load)
    for (int i=0; i<count; i++)
    {
        const path_t& imagepath = ltp->input_files[i];

        unsigned char* pixeldata = 0;
        int w;
        int h;
        int c;

#if _WIN32
        pixeldata = wic_decode_image(imagepath.c_str(), &w, &h, &c);
#else // _WIN32
        pixeldata = stbi_load(imagepath.c_str(), &w, &h, &c, 3);
#endif // _WIN32
        if (pixeldata)
        {
            Task v;
            v.id = i;
            v.inpath = imagepath;
            v.outpath = ltp->output_files[i];

            v.inimage = ncnn::Mat(w, h, (void*)pixeldata, (size_t)3, 3);
            v.outimage = ncnn::Mat(w * scale, h * scale, (size_t)3u, 3);

            toproc.put(v);
        }
        else
        {
#if _WIN32
            fwprintf(stderr, L"decode image %ls failed\n", imagepath.c_str());
#else // _WIN32
            fprintf(stderr, "decode image %s failed\n", imagepath.c_str());
#endif // _WIN32
        }
    }

    return 0;
}

class ProcThreadParams
{
public:
    const Waifu2x* waifu2x;
};

void* proc(void* args)
{
    const ProcThreadParams* ptp = (const ProcThreadParams*)args;
    const Waifu2x* waifu2x = ptp->waifu2x;

    for (;;)
    {
        Task v;

        toproc.get(v);

        if (v.id == -233)
            break;

        waifu2x->process(v.inimage, v.outimage);

        tosave.put(v);
    }

    return 0;
}

class SaveThreadParams
{
public:
    int verbose;
};

void* save(void* args)
{
    const SaveThreadParams* stp = (const SaveThreadParams*)args;
    const int verbose = stp->verbose;

    for (;;)
    {
        Task v;

        tosave.get(v);

        if (v.id == -233)
            break;

        // free input pixel data
        {
            unsigned char* pixeldata = (unsigned char*)v.inimage.data;
#if _WIN32
            free(pixeldata);
#else
            stbi_image_free(pixeldata);
#endif
        }

#if _WIN32
        int success = wic_encode_image(v.outpath.c_str(), v.outimage.w, v.outimage.h, 3, v.outimage.data);
#else
        int success = stbi_write_png(v.outpath.c_str(), v.outimage.w, v.outimage.h, 3, v.outimage.data, 0);
#endif
        if (success)
        {
            if (verbose)
            {
#if _WIN32
                fwprintf(stderr, L"%ls -> %ls done\n", v.inpath.c_str(), v.outpath.c_str());
#else
                fprintf(stderr, "%s -> %s done\n", v.inpath.c_str(), v.outpath.c_str());
#endif
            }
        }
        else
        {
#if _WIN32
            fwprintf(stderr, L"encode image %ls failed\n", v.outpath.c_str());
#else
            fprintf(stderr, "encode image %s failed\n", v.outpath.c_str());
#endif
        }
    }

    return 0;
}


#if _WIN32
int wmain(int argc, wchar_t** argv)
#else
int main(int argc, char** argv)
#endif
{
    path_t inputpath;
    path_t outputpath;
    int noise = 0;
    int scale = 2;
    int tilesize = 400;
    path_t model = PATHSTR("models-cunet");
    int gpuid = 0;
    int jobs_load = 1;
    int jobs_proc = 2;
    int jobs_save = 2;
    int verbose = 0;

#if _WIN32
    wchar_t opt;
    while ((opt = getopt(argc, argv, L"i:o:n:s:t:m:g:j:vh")) != (wchar_t)-1)
    {
        switch (opt)
        {
        case L'i':
            inputpath = optarg;
            break;
        case L'o':
            outputpath = optarg;
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
        case L'j':
            swscanf(optarg, L"%d:%d:%d", &jobs_load, &jobs_proc, &jobs_save);
            break;
        case L'v':
            verbose = 1;
            break;
        case L'h':
        default:
            print_usage();
            return -1;
        }
    }
#else // _WIN32
    int opt;
    while ((opt = getopt(argc, argv, "i:o:n:s:t:m:g:j:vh")) != -1)
    {
        switch (opt)
        {
        case 'i':
            inputpath = optarg;
            break;
        case 'o':
            outputpath = optarg;
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
        case 'j':
            sscanf(optarg, "%d:%d:%d", &jobs_load, &jobs_proc, &jobs_save);
            break;
        case 'v':
            verbose = 1;
            break;
        case 'h':
        default:
            print_usage();
            return -1;
        }
    }
#endif // _WIN32

    if (inputpath.empty() || outputpath.empty())
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

    if (jobs_load < 1 || jobs_proc < 1 || jobs_save < 1)
    {
        fprintf(stderr, "invalid thread count argument\n");
        return -1;
    }

    // collect input and output filepath
    std::vector<path_t> input_files;
    std::vector<path_t> output_files;
    {
        if (path_is_directory(inputpath) && path_is_directory(outputpath))
        {
            std::vector<path_t> filenames;
            int lr = list_directory(inputpath, filenames);
            if (lr != 0)
                return -1;

            const int count = filenames.size();
            input_files.resize(count);
            output_files.resize(count);
            for (int i=0; i<count; i++)
            {
                input_files[i] = inputpath + PATHSTR('/') + filenames[i];
                output_files[i] = outputpath + PATHSTR('/') + filenames[i] + PATHSTR(".png");
            }
        }
        else if (!path_is_directory(inputpath) && !path_is_directory(outputpath))
        {
            input_files.push_back(inputpath);
            output_files.push_back(outputpath);
        }
        else
        {
            fprintf(stderr, "inputpath and outputpath must be either file or directory at the same time\n");
            return -1;
        }
    }

    int prepadding = 0;

    if (model.find(PATHSTR("models-cunet")) != path_t::npos)
    {
        if (noise == -1)
        {
            prepadding = 18;
        }
        else if (scale == 1)
        {
            prepadding = 28;
        }
        else if (scale == 2)
        {
            prepadding = 18;
        }
    }
    else if (model.find(PATHSTR("models-upconv_7_anime_style_art_rgb")) != path_t::npos)
    {
        prepadding = 7;
    }
    else if (model.find(PATHSTR("models-upconv_7_photo")) != path_t::npos)
    {
        prepadding = 7;
    }
    else
    {
        fprintf(stderr, "unknown model dir type\n");
        return -1;
    }

#if _WIN32
    wchar_t parampath[256];
    wchar_t modelpath[256];
    if (noise == -1)
    {
        swprintf(parampath, 256, L"%s/scale2.0x_model.param", model.c_str());
        swprintf(modelpath, 256, L"%s/scale2.0x_model.bin", model.c_str());
    }
    else if (scale == 1)
    {
        swprintf(parampath, 256, L"%s/noise%d_model.param", model.c_str(), noise);
        swprintf(modelpath, 256, L"%s/noise%d_model.bin", model.c_str(), noise);
    }
    else if (scale == 2)
    {
        swprintf(parampath, 256, L"%s/noise%d_scale2.0x_model.param", model.c_str(), noise);
        swprintf(modelpath, 256, L"%s/noise%d_scale2.0x_model.bin", model.c_str(), noise);
    }
#else
    char parampath[256];
    char modelpath[256];
    if (noise == -1)
    {
        sprintf(parampath, "%s/scale2.0x_model.param", model.c_str());
        sprintf(modelpath, "%s/scale2.0x_model.bin", model.c_str());
    }
    else if (scale == 1)
    {
        sprintf(parampath, "%s/noise%d_model.param", model.c_str(), noise);
        sprintf(modelpath, "%s/noise%d_model.bin", model.c_str(), noise);
    }
    else if (scale == 2)
    {
        sprintf(parampath, "%s/noise%d_scale2.0x_model.param", model.c_str(), noise);
        sprintf(modelpath, "%s/noise%d_scale2.0x_model.bin", model.c_str(), noise);
    }
#endif

#if _WIN32
    CoInitializeEx(NULL, COINIT_MULTITHREADED);
#endif

    ncnn::create_gpu_instance();

    int cpu_count = std::max(1, ncnn::get_cpu_count());
    jobs_load = std::min(jobs_load, cpu_count);
    jobs_save = std::min(jobs_save, cpu_count);

    int gpu_count = ncnn::get_gpu_count();
    if (gpuid < 0 || gpuid >= gpu_count)
    {
        fprintf(stderr, "invalid gpu device\n");

        ncnn::destroy_gpu_instance();
        return -1;
    }

    int gpu_queue_count = ncnn::get_gpu_info(gpuid).compute_queue_count;
    jobs_proc = std::min(jobs_proc, gpu_queue_count);

    {
        Waifu2x waifu2x(gpuid);

        waifu2x.load(parampath, modelpath);

        waifu2x.noise = noise;
        waifu2x.scale = scale;
        waifu2x.tilesize = tilesize;
        waifu2x.prepadding = prepadding;

        // main routine
        {
            // load image
            LoadThreadParams ltp;
            ltp.scale = scale;
            ltp.jobs_load = jobs_load;
            ltp.input_files = input_files;
            ltp.output_files = output_files;

            ncnn::Thread load_thread(load, (void*)&ltp);

            // waifu2x proc
            ProcThreadParams ptp;
            ptp.waifu2x = &waifu2x;

            std::vector<ncnn::Thread*> proc_threads(jobs_proc);
            for (int i=0; i<jobs_proc; i++)
            {
                proc_threads[i] = new ncnn::Thread(proc, (void*)&ptp);
            }

            // save image
            SaveThreadParams stp;
            stp.verbose = verbose;

            std::vector<ncnn::Thread*> save_threads(jobs_save);
            for (int i=0; i<jobs_save; i++)
            {
                save_threads[i] = new ncnn::Thread(save, (void*)&stp);
            }

            // end
            load_thread.join();

            Task end;
            end.id = -233;

            for (int i=0; i<jobs_proc; i++)
            {
                toproc.put(end);
            }

            for (int i=0; i<jobs_proc; i++)
            {
                proc_threads[i]->join();
                delete proc_threads[i];
            }

            for (int i=0; i<jobs_save; i++)
            {
                tosave.put(end);
            }

            for (int i=0; i<jobs_save; i++)
            {
                save_threads[i]->join();
                delete save_threads[i];
            }
        }
    }

    ncnn::destroy_gpu_instance();

    return 0;
}
