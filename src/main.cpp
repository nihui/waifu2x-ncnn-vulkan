// waifu2x implemented with ncnn library

#include <stdio.h>
#include <algorithm>
#include <queue>
#include <vector>
#include <functional>
#include <condition_variable>
#include <clocale>

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
#define STBI_NO_STDIO
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif // _WIN32
#include "webp_image.h"
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

#if _WIN32
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

static std::vector<int> parse_optarg_int_array(const wchar_t* optarg)
{
    std::vector<int> array;
    array.push_back(_wtoi(optarg));

    const wchar_t* p = wcschr(optarg, L',');
    while (p)
    {
        p++;
        array.push_back(_wtoi(p));
        p = wcschr(p, L',');
    }

    return array;
}
#else // _WIN32
#include <unistd.h> // getopt()

static std::vector<int> parse_optarg_int_array(const char* optarg)
{
    std::vector<int> array;
    array.push_back(atoi(optarg));

    const char* p = strchr(optarg, ',');
    while (p)
    {
        p++;
        array.push_back(atoi(p));
        p = strchr(p, ',');
    }

    return array;
}
#endif // _WIN32

// ncnn
#include "cpu.h"
#include "gpu.h"
#include "platform.h"

#include "waifu2x.h"

#include "filesystem_utils.h"

static void print_usage()
{
    fprintf(stdout, "Usage: waifu2x-ncnn-vulkan -i infile -o outfile [options]...\n");
    fprintf(stdout, "       waifu2x-ncnn-vulkan -a invideo -b outvideo [options]...\n\n");
    fprintf(stdout, "  -h                   show this help\n");
    fprintf(stdout, "  -v                   verbose output\n");
    fprintf(stdout, "  -i input-path        input image path (jpg/png/webp) or directory\n");
    fprintf(stdout, "  -o output-path       output image path (jpg/png/webp) or directory\n");
    fprintf(stdout, "  -a video input       input video path\n");
    fprintf(stdout, "  -b video output      output video path\n");
    fprintf(stdout, "  -n noise-level       denoise level (-1/0/1/2/3, default=0)\n");
    fprintf(stdout, "  -s scale             upscale ratio (1/2, default=2)\n");
    fprintf(stdout, "  -t tile-size         tile size (>=32/0=auto, default=0) can be 0,0,0 for multi-gpu\n");
    fprintf(stdout, "  -m model-path        waifu2x model path (default=models-cunet)\n");
    fprintf(stdout, "  -g gpu-id            gpu device to use (default=auto) can be 0,1,2 for multi-gpu\n");
    fprintf(stdout, "  -j load:proc:save    thread count for load/proc/save (default=1:2:2) can be 1:2,2,2:2 for multi-gpu\n");
    fprintf(stdout, "  -x                   enable tta mode\n");
    fprintf(stdout, "  -f format            output image format (jpg/png/webp, default=ext/png)\n");
}

class Task
{
public:
    int id;
    int webp;

    path_t inpath;
    path_t outpath;

    ncnn::Mat inimage;
    ncnn::Mat outimage;

    bool video_mode;
    std::function<void(cv::Mat& scaled_frame, int frame_index)> frame_scaled_cb;
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
    bool video_mode;

    // session data
    std::vector<path_t> input_files;
    std::vector<path_t> output_files;

    path_t input_video;
    path_t output_video;

    int total_jobs_proc;
    int jobs_save;
    std::vector<ncnn::Thread*> proc_threads;
    std::vector<ncnn::Thread*> save_threads;
};

void* load_video(const LoadThreadParams* ltp)
{
    cv::VideoCapture video(ltp->input_video);
    if (video.isOpened())
    {
        int frame_index = 0;
        uint64_t total_frames = video.get(cv::CAP_PROP_FRAME_COUNT);
        uint64_t processed_frames = 0;
        int scale = ltp->scale;
        std::mutex m;
        std::condition_variable condition;

        bool done;
        cv::VideoWriter writer(ltp->output_video.c_str(),
                               cv::VideoWriter::fourcc('h', 'v', 'c', '1'),
                               video.get(cv::CAP_PROP_FPS),
                               cv::Size(
                                       (int)(video.get(cv::CAP_PROP_FRAME_WIDTH) * scale),
                                       (int)(video.get(cv::CAP_PROP_FRAME_HEIGHT) * scale)),true);
        auto cb = [&writer, &processed_frames, &done, &condition, total_frames, ltp](cv::Mat& scaled_frame, int frame_index) {
            fprintf(stderr, "video frame %d processed! [%llu/%llu]\n", frame_index, processed_frames + 1, total_frames);
            writer.write(scaled_frame);
            processed_frames++;
            if (processed_frames == total_frames)
            {
                fprintf(stderr, "all frames processed\n");
                Task end;
                end.id = -233;

                for (int i=0; i<ltp->total_jobs_proc; i++)
                {
                    toproc.put(end);
                }

                for (int i=0; i<ltp->jobs_save; i++)
                {
                    tosave.put(end);
                }

                done = true;
                condition.notify_one();
                fprintf(stderr, "notified\n");
            }
        };
        while (true) {
            cv::Mat frame;
            video >> frame;
            if (frame.empty()) {
                break;
            }
            Task v;
            v.id = frame_index;
            frame_index++;
            v.webp = false;
            v.video_mode = true;
            v.frame_scaled_cb = cb;

            int w = frame.size().width;
            int h = frame.size().height;
            int c = frame.channels();
            v.inimage = ncnn::Mat(w, h, (void*)frame.data, (size_t)c, c);
            v.outimage = ncnn::Mat(w * scale, h * scale, (size_t)c, c);

            toproc.put(v);
        }

        video.release();

        {
            std::unique_lock<std::mutex> lk(m);
            condition.wait(lk, [&done]{return done;});
        }

        for (int i=0; i<ltp->total_jobs_proc; i++)
        {
            ltp->proc_threads[i]->join();
            delete ltp->proc_threads[i];
        }
        for (int i=0; i<ltp->jobs_save; i++)
        {
            ltp->save_threads[i]->join();
            delete ltp->save_threads[i];
        }
    }
    else
    {
#if _WIN32
        fwprintf(stderr, L"video %ls cannot be opened ! skipped !\n", ltp->input_video.c_str());
#else // _WIN32
        fprintf(stderr, "video %s cannot be opened ! skipped !\n", ltp->input_video.c_str());
#endif // _WIN32
    }

    return 0;
}

void* load(void* args)
{
    const LoadThreadParams* ltp = (const LoadThreadParams*)args;
    if (ltp->video_mode)
    {
        return load_video(ltp);
    }
    const int count = ltp->input_files.size();
    const int scale = ltp->scale;

    #pragma omp parallel for schedule(static,1) num_threads(ltp->jobs_load)
    for (int i=0; i<count; i++)
    {
        const path_t& imagepath = ltp->input_files[i];

        int webp = 0;

        unsigned char* pixeldata = 0;
        int w;
        int h;
        int c;

#if _WIN32
        FILE* fp = _wfopen(imagepath.c_str(), L"rb");
#else
        FILE* fp = fopen(imagepath.c_str(), "rb");
#endif
        if (fp)
        {
            // read whole file
            unsigned char* filedata = 0;
            int length = 0;
            {
                fseek(fp, 0, SEEK_END);
                length = ftell(fp);
                rewind(fp);
                filedata = (unsigned char*)malloc(length);
                if (filedata)
                {
                    fread(filedata, 1, length, fp);
                }
                fclose(fp);
            }

            if (filedata)
            {
                pixeldata = webp_load(filedata, length, &w, &h, &c);
                if (pixeldata)
                {
                    webp = 1;
                }
                else
                {
                    // not webp, try jpg png etc.
#if _WIN32
                    pixeldata = wic_decode_image(imagepath.c_str(), &w, &h, &c);
#else // _WIN32
                    pixeldata = stbi_load_from_memory(filedata, length, &w, &h, &c, 0);
                    if (pixeldata)
                    {
                        // stb_image auto channel
                        if (c == 1)
                        {
                            // grayscale -> rgb
                            stbi_image_free(pixeldata);
                            pixeldata = stbi_load_from_memory(filedata, length, &w, &h, &c, 3);
                            c = 3;
                        }
                        else if (c == 2)
                        {
                            // grayscale + alpha -> rgba
                            stbi_image_free(pixeldata);
                            pixeldata = stbi_load_from_memory(filedata, length, &w, &h, &c, 4);
                            c = 4;
                        }
                    }
#endif // _WIN32
                }

                free(filedata);
            }
        }
        if (pixeldata)
        {
            Task v;
            v.id = i;
            v.webp = webp;
            v.inpath = imagepath;
            v.outpath = ltp->output_files[i];

            v.inimage = ncnn::Mat(w, h, (void*)pixeldata, (size_t)c, c);
            v.outimage = ncnn::Mat(w * scale, h * scale, (size_t)c, c);

            path_t ext = get_file_extension(v.outpath);
            if (c == 4 && (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") || ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG")))
            {
                path_t output_filename2 = ltp->output_files[i] + PATHSTR(".png");
                v.outpath = output_filename2;
#if _WIN32
                fwprintf(stderr, L"image %ls has alpha channel ! %ls will output %ls\n", imagepath.c_str(), imagepath.c_str(), output_filename2.c_str());
#else // _WIN32
                fprintf(stderr, "image %s has alpha channel ! %s will output %s\n", imagepath.c_str(), imagepath.c_str(), output_filename2.c_str());
#endif // _WIN32
            }

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

        if (v.video_mode)
        {
            cv::Mat scaled_frame(v.outimage.h, v.outimage.w, CV_8UC3, v.outimage.data);
            v.frame_scaled_cb(scaled_frame, v.id);
        }
        else
        {
            // free input pixel data
            {
                unsigned char* pixeldata = (unsigned char*)v.inimage.data;
                if (v.webp == 1)
                {
                    free(pixeldata);
                }
                else
                {
#if _WIN32
                    free(pixeldata);
#else
                    stbi_image_free(pixeldata);
#endif
                }
            }

            int success = 0;

            path_t ext = get_file_extension(v.outpath);

            if (ext == PATHSTR("webp") || ext == PATHSTR("WEBP"))
            {
                success = webp_save(v.outpath.c_str(), v.outimage.w, v.outimage.h, v.outimage.elempack, (const unsigned char*)v.outimage.data);
            }
            else if (ext == PATHSTR("png") || ext == PATHSTR("PNG"))
            {
#if _WIN32
                success = wic_encode_image(v.outpath.c_str(), v.outimage.w, v.outimage.h, v.outimage.elempack, v.outimage.data);
#else
                success = stbi_write_png(v.outpath.c_str(), v.outimage.w, v.outimage.h, v.outimage.elempack, v.outimage.data, 0);
#endif
            }
            else if (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") || ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG"))
            {
#if _WIN32
                success = wic_encode_jpeg_image(v.outpath.c_str(), v.outimage.w, v.outimage.h, v.outimage.elempack, v.outimage.data);
#else
                success = stbi_write_jpg(v.outpath.c_str(), v.outimage.w, v.outimage.h, v.outimage.elempack, v.outimage.data, 100);
#endif
            }
            if (success)
            {
                if (verbose)
                {
#if _WIN32
                    fwprintf(stdout, L"%ls -> %ls done\n", v.inpath.c_str(), v.outpath.c_str());
#else
                    fprintf(stdout, "%s -> %s done\n", v.inpath.c_str(), v.outpath.c_str());
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
    }

    return 0;
}

int load_model_param(path_t model, int noise, int scale, path_t &out_paramfullpath, path_t &out_modelfullpath, int &out_prepadding)
{
    out_prepadding = 0;
    fprintf(stderr, "model: %s\n", model.c_str());

    if (model.find(PATHSTR("models-cunet")) != path_t::npos)
    {
        if (noise == -1)
        {
            out_prepadding = 18;
        }
        else if (scale == 1)
        {
            out_prepadding = 28;
        }
        else if (scale == 2)
        {
            out_prepadding = 18;
        }
    }
    else if (model.find(PATHSTR("models-upconv_7_anime_style_art_rgb")) != path_t::npos)
    {
        out_prepadding = 7;
    }
    else if (model.find(PATHSTR("models-upconv_7_photo")) != path_t::npos)
    {
        out_prepadding = 7;
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

    out_paramfullpath = sanitize_filepath(parampath);
    out_modelfullpath = sanitize_filepath(modelpath);

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

    path_t video_inputpath;
    path_t video_outputpath;

    int noise = 0;
    int scale = 2;
    std::vector<int> tilesize;
    path_t model = PATHSTR("models-cunet");
    std::vector<int> gpuid;
    int jobs_load = 1;
    std::vector<int> jobs_proc;
    int jobs_save = 2;
    int verbose = 0;
    int tta_mode = 0;
    path_t format = PATHSTR("png");

#if _WIN32
    setlocale(LC_ALL, "");
    wchar_t opt;
    while ((opt = getopt(argc, argv, L"i:o:a:b:n:s:t:m:g:j:f:vxh")) != (wchar_t)-1)
    {
        switch (opt)
        {
        case L'i':
            inputpath = optarg;
            break;
        case L'o':
            outputpath = optarg;
            break;
        case L'a':
            video_inputpath = optarg;
            break;
        case L'b':
            video_outputpath = optarg;
            break;
        case L'n':
            noise = _wtoi(optarg);
            break;
        case L's':
            scale = _wtoi(optarg);
            break;
        case L't':
            tilesize = parse_optarg_int_array(optarg);
            break;
        case L'm':
            model = optarg;
            break;
        case L'g':
            gpuid = parse_optarg_int_array(optarg);
            break;
        case L'j':
            swscanf(optarg, L"%d:%*[^:]:%d", &jobs_load, &jobs_save);
            jobs_proc = parse_optarg_int_array(wcschr(optarg, L':') + 1);
            break;
        case L'f':
            format = optarg;
            break;
        case L'v':
            verbose = 1;
            break;
        case L'x':
            tta_mode = 1;
            break;
        case L'h':
        default:
            print_usage();
            return -1;
        }
    }
#else // _WIN32
    int opt;
    while ((opt = getopt(argc, argv, "i:o:a:b:n:s:t:m:g:j:f:vxh")) != -1)
    {
        switch (opt)
        {
        case 'i':
            inputpath = optarg;
            break;
        case 'o':
            outputpath = optarg;
            break;
        case 'a':
            video_inputpath = optarg;
            break;
        case 'b':
            video_outputpath = optarg;
            break;
        case 'n':
            noise = atoi(optarg);
            break;
        case 's':
            scale = atoi(optarg);
            break;
        case 't':
            tilesize = parse_optarg_int_array(optarg);
            break;
        case 'm':
            model = optarg;
            break;
        case 'g':
            gpuid = parse_optarg_int_array(optarg);
            break;
        case 'j':
            sscanf(optarg, "%d:%*[^:]:%d", &jobs_load, &jobs_save);
            jobs_proc = parse_optarg_int_array(strchr(optarg, ':') + 1);
            break;
        case 'f':
            format = optarg;
            break;
        case 'v':
            verbose = 1;
            break;
        case 'x':
            tta_mode = 1;
            break;
        case 'h':
        default:
            print_usage();
            return -1;
        }
    }
#endif // _WIN32

    bool video_mode = false;
    bool empty_input = false;
    if (inputpath.empty() || outputpath.empty())
    {
        empty_input = true;
        if (!video_inputpath.empty() && !video_outputpath.empty())
        {
            empty_input = false;
            video_mode = true;
        }
    }

    if (empty_input)
    {
        print_usage();
        return -1;
    }

    if (noise < -1 || noise > 3)
    {
        fprintf(stderr, "invalid noise argument\n");
        return -1;
    }

    if (scale < 1 || scale > 2)
    {
        fprintf(stderr, "invalid scale argument\n");
        return -1;
    }

    if (tilesize.size() != (gpuid.empty() ? 1 : gpuid.size()) && !tilesize.empty())
    {
        fprintf(stderr, "invalid tilesize argument\n");
        return -1;
    }

    for (int i=0; i<(int)tilesize.size(); i++)
    {
        if (tilesize[i] != 0 && tilesize[i] < 32)
        {
            fprintf(stderr, "invalid tilesize argument\n");
            return -1;
        }
    }

    if (jobs_load < 1 || jobs_save < 1)
    {
        fprintf(stderr, "invalid thread count argument\n");
        return -1;
    }

    if (jobs_proc.size() != (gpuid.empty() ? 1 : gpuid.size()) && !jobs_proc.empty())
    {
        fprintf(stderr, "invalid jobs_proc thread count argument\n");
        return -1;
    }

    for (int i=0; i<(int)jobs_proc.size(); i++)
    {
        if (jobs_proc[i] < 1)
        {
            fprintf(stderr, "invalid jobs_proc thread count argument\n");
            return -1;
        }
    }

    int prepadding = 0;
    path_t paramfullpath;
    path_t modelfullpath;

    // image mode
    // collect input and output filepath
    std::vector<path_t> input_files;
    std::vector<path_t> output_files;

    if (!video_mode)
    {
        if (!path_is_directory(outputpath))
        {
            // guess format from outputpath no matter what format argument specified
            path_t ext = get_file_extension(outputpath);

            if (ext == PATHSTR("png") || ext == PATHSTR("PNG"))
            {
                format = PATHSTR("png");
            }
            else if (ext == PATHSTR("webp") || ext == PATHSTR("WEBP"))
            {
                format = PATHSTR("webp");
            }
            else if (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") || ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG"))
            {
                format = PATHSTR("jpg");
            }
            else
            {
                fprintf(stderr, "invalid outputpath extension type\n");
                return -1;
            }
        }

        if (format != PATHSTR("png") && format != PATHSTR("webp") && format != PATHSTR("jpg"))
        {
            fprintf(stderr, "invalid format argument\n");
            return -1;
        }

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

                path_t last_filename;
                path_t last_filename_noext;
                for (int i=0; i<count; i++)
                {
                    path_t filename = filenames[i];
                    path_t filename_noext = get_file_name_without_extension(filename);
                    path_t output_filename = filename_noext + PATHSTR('.') + format;

                    // filename list is sorted, check if output image path conflicts
                    if (filename_noext == last_filename_noext)
                    {
                        path_t output_filename2 = filename + PATHSTR('.') + format;
#if _WIN32
                        fwprintf(stderr, L"both %ls and %ls output %ls ! %ls will output %ls\n", filename.c_str(), last_filename.c_str(), output_filename.c_str(), filename.c_str(), output_filename2.c_str());
#else
                        fprintf(stderr, "both %s and %s output %s ! %s will output %s\n", filename.c_str(), last_filename.c_str(), output_filename.c_str(), filename.c_str(), output_filename2.c_str());
#endif
                        output_filename = output_filename2;
                    }
                    else
                    {
                        last_filename = filename;
                        last_filename_noext = filename_noext;
                    }

                    input_files[i] = inputpath + PATHSTR('/') + filename;
                    output_files[i] = outputpath + PATHSTR('/') + output_filename;
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
    }

    int ret = load_model_param(model, noise, scale, paramfullpath, modelfullpath, prepadding);
    if (ret != 0)
    {
        return ret;
    }

#if _WIN32
    CoInitializeEx(NULL, COINIT_MULTITHREADED);
#endif

    // debug
    setenv("VK_ICD_FILENAMES", "/Users/Cocoa/YouTube/MoltenVK_icd.json", 1);
    ncnn::create_gpu_instance();

    if (gpuid.empty())
    {
        gpuid.push_back(ncnn::get_default_gpu_index());
    }

    const int use_gpu_count = (int)gpuid.size();

    if (jobs_proc.empty())
    {
        jobs_proc.resize(use_gpu_count, 2);
    }

    if (tilesize.empty())
    {
        tilesize.resize(use_gpu_count, 0);
    }

    int cpu_count = std::max(1, ncnn::get_cpu_count());
    jobs_load = std::min(jobs_load, cpu_count);
    jobs_save = std::min(jobs_save, cpu_count);

    int gpu_count = ncnn::get_gpu_count();
    for (int i=0; i<use_gpu_count; i++)
    {
        if (gpuid[i] < 0 || gpuid[i] >= gpu_count)
        {
            fprintf(stderr, "invalid gpu device\n");

            ncnn::destroy_gpu_instance();
            return -1;
        }
    }

    int total_jobs_proc = 0;
    for (int i=0; i<use_gpu_count; i++)
    {
        int gpu_queue_count = ncnn::get_gpu_info(gpuid[i]).compute_queue_count;
        jobs_proc[i] = std::min(jobs_proc[i], gpu_queue_count);
        total_jobs_proc += jobs_proc[i];
    }

    for (int i=0; i<use_gpu_count; i++)
    {
        if (tilesize[i] != 0)
            continue;

        uint32_t heap_budget = ncnn::get_gpu_device(gpuid[i])->get_heap_budget();

        // more fine-grained tilesize policy here
        if (model.find(PATHSTR("models-cunet")) != path_t::npos)
        {
            if (heap_budget > 2600)
                tilesize[i] = 400;
            else if (heap_budget > 740)
                tilesize[i] = 200;
            else if (heap_budget > 250)
                tilesize[i] = 100;
            else
                tilesize[i] = 32;
        }
        else if (model.find(PATHSTR("models-upconv_7_anime_style_art_rgb")) != path_t::npos
            || model.find(PATHSTR("models-upconv_7_photo")) != path_t::npos)
        {
            if (heap_budget > 1900)
                tilesize[i] = 400;
            else if (heap_budget > 550)
                tilesize[i] = 200;
            else if (heap_budget > 190)
                tilesize[i] = 100;
            else
                tilesize[i] = 32;
        }
    }

    {
        std::vector<Waifu2x*> waifu2x(use_gpu_count);

        for (int i=0; i<use_gpu_count; i++)
        {
            waifu2x[i] = new Waifu2x(gpuid[i], tta_mode);

            waifu2x[i]->load(paramfullpath, modelfullpath);

            waifu2x[i]->noise = noise;
            waifu2x[i]->scale = scale;
            waifu2x[i]->tilesize = tilesize[i];
            waifu2x[i]->prepadding = prepadding;
        }

        // main routine
        {
            // load param
            LoadThreadParams ltp;
            ltp.scale = scale;
            ltp.video_mode = video_mode;
            if (!video_mode)
            {
                ltp.jobs_load = jobs_load;
                ltp.input_files = input_files;
                ltp.output_files = output_files;
            }
            else
            {
                ltp.jobs_load = jobs_load;
                ltp.input_video = video_inputpath;
                ltp.output_video = video_outputpath;
                ltp.jobs_save = jobs_save;
                ltp.total_jobs_proc = total_jobs_proc;
            }
            ncnn::Thread load_thread(load, (void*)&ltp);

            // waifu2x proc
            std::vector<ProcThreadParams> ptp(use_gpu_count);
            for (int i=0; i<use_gpu_count; i++)
            {
                ptp[i].waifu2x = waifu2x[i];
            }

            std::vector<ncnn::Thread*> proc_threads(total_jobs_proc);
            {
                int total_jobs_proc_id = 0;
                for (int i=0; i<use_gpu_count; i++)
                {
                    for (int j=0; j<jobs_proc[i]; j++)
                    {
                        proc_threads[total_jobs_proc_id++] = new ncnn::Thread(proc, (void*)&ptp[i]);
                    }
                }
            }

            // save image
            SaveThreadParams stp;
            stp.verbose = verbose;

            std::vector<ncnn::Thread*> save_threads(jobs_save);
            for (int i=0; i<jobs_save; i++)
            {
                save_threads[i] = new ncnn::Thread(save, (void*)&stp);
            }

            if (video_mode) {
                ltp.proc_threads = proc_threads;
                ltp.save_threads = save_threads;
            }

            // end
            if (!video_mode)
            {
                load_thread.join();

                Task end;
                end.id = -233;

                for (int i=0; i<total_jobs_proc; i++)
                {
                    toproc.put(end);
                }

                for (int i=0; i<total_jobs_proc; i++)
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
            else
            {
                load_thread.join();
            }
        }

        for (int i=0; i<use_gpu_count; i++)
        {
            delete waifu2x[i];
        }
        waifu2x.clear();
    }

    ncnn::destroy_gpu_instance();

    return 0;
}
