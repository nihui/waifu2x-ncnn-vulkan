#ifndef DLL_EXPORT
#ifdef _WIN32
#ifdef _WINDLL
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif
#else
#define DLL_EXPORT __attribute__((visibility("default")))
#endif
#endif

#include "string"
#include "waifu2x.h"

extern "C" DLL_EXPORT void* create_module(const wchar_t* model, const wchar_t* bin)
{
    ncnn::create_gpu_instance();

    std::wstring modelw(model);
    std::wstring binw(bin);

    if (model)
    {
        auto waifu2x = new Waifu2x(0, 0);
        waifu2x->load(modelw, binw);
        waifu2x->noise = -1;
        waifu2x->scale = 2;
        waifu2x->tilesize = 200;
        waifu2x->prepadding = 7;

        if (modelw.find(L"models-cunet") != modelw.npos)
        {
            if (waifu2x->noise == -1)
            {
                waifu2x->prepadding = 18;
            }
            else if (waifu2x->scale == 1)
            {
                waifu2x->prepadding = 28;
            }
            else if (waifu2x->scale == 2)
            {
                waifu2x->prepadding = 18;
            }
        }
        return waifu2x;
    }
    return nullptr;
}

extern "C" DLL_EXPORT void run_module(void* module, int w, int h, int c, const char* src, char* dst)
{
    auto waifu2x = (Waifu2x*)module;
    if (waifu2x)
    {
        ncnn::Mat m(w, h, (void*)src, 3u, 3);
        ncnn::Mat m1(w * 2, h * 2, (void*)dst, 3u, 3);
        waifu2x->process(m, m1);
    }
}

extern "C" DLL_EXPORT void destroy_module(void* module)
{
    auto waifu2x = (Waifu2x*)module;
    delete waifu2x;
    ncnn::destroy_gpu_instance();
}