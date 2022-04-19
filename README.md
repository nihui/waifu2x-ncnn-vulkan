# waifu2x ncnn Vulkan

![CI](https://github.com/nihui/waifu2x-ncnn-vulkan/workflows/CI/badge.svg)
![download](https://img.shields.io/github/downloads/nihui/waifu2x-ncnn-vulkan/total.svg)

ncnn implementation of waifu2x converter. Runs fast on Intel / AMD / Nvidia / Apple-Silicon with Vulkan API.

waifu2x-ncnn-vulkan uses [ncnn project](https://github.com/Tencent/ncnn) as the universal neural network inference framework.

## [Download](https://github.com/nihui/waifu2x-ncnn-vulkan/releases)

Download Windows/Linux/MacOS Executable for Intel/AMD/Nvidia GPU

**https://github.com/nihui/waifu2x-ncnn-vulkan/releases**

This package includes all the binaries and models required. It is portable, so no CUDA or Caffe runtime environment is needed :)

## Usages

### Example Command

```shell
waifu2x-ncnn-vulkan.exe -i input.jpg -o output.png -n 2 -s 2
```

### Full Usages

```console
Usage: waifu2x-ncnn-vulkan -i infile -o outfile [options]...

  -h                   show this help
  -v                   verbose output
  -i input-path        input image path (jpg/png/webp) or directory
  -o output-path       output image path (jpg/png/webp) or directory
  -n noise-level       denoise level (-1/0/1/2/3, default=0)
  -s scale             upscale ratio (1/2/4/8/16/32, default=2)
  -t tile-size         tile size (>=32/0=auto, default=0) can be 0,0,0 for multi-gpu
  -m model-path        waifu2x model path (default=models-cunet)
  -g gpu-id            gpu device to use (-1=cpu, default=auto) can be 0,1,2 for multi-gpu
  -j load:proc:save    thread count for load/proc/save (default=1:2:2) can be 1:2,2,2:2 for multi-gpu
  -x                   enable tta mode
  -f format            output image format (jpg/png/webp, default=ext/png)
```

- `input-path` and `output-path` accept either file path or directory path
- `noise-level` = noise level, large value means strong denoise effect, -1 = no effect
- `scale` = scale level, 1 = no scaling, 2 = upscale 2x
- `tile-size` = tile size, use smaller value to reduce GPU memory usage, default selects automatically
- `load:proc:save` = thread count for the three stages (image decoding + waifu2x upscaling + image encoding), using larger values may increase GPU usage and consume more GPU memory. You can tune this configuration with "4:4:4" for many small-size images, and "2:2:2" for large-size images. The default setting usually works fine for most situations. If you find that your GPU is hungry, try increasing thread count to achieve faster processing.
- `format` = the format of the image to be output, png is better supported, however webp generally yields smaller file sizes, both are losslessly encoded

If you encounter a crash or error, try upgrading your GPU driver:

- Intel: https://downloadcenter.intel.com/product/80939/Graphics-Drivers
- AMD: https://www.amd.com/en/support
- NVIDIA: https://www.nvidia.com/Download/index.aspx

## Build from Source

1. Download and setup the Vulkan SDK from https://vulkan.lunarg.com/
  - For Linux distributions, you can either get the essential build requirements from package manager
```shell
dnf install vulkan-headers vulkan-loader-devel
```
```shell
apt-get install libvulkan-dev
```
```shell
pacman -S vulkan-headers vulkan-icd-loader
```

2. Clone this project with all submodules

```shell
git clone https://github.com/nihui/waifu2x-ncnn-vulkan.git
cd waifu2x-ncnn-vulkan
git submodule update --init --recursive
```

3. Build with CMake
  - You can pass -DUSE_STATIC_MOLTENVK=ON option to avoid linking the vulkan loader library on MacOS

```shell
mkdir build
cd build
cmake ../src
cmake --build . -j 4
```

## Speed Comparison with waifu2x-caffe-cui

### Environment

- Windows 10 1809
- AMD R7-1700
- Nvidia GTX-1070
- Nvidia driver 419.67
- CUDA 10.1.105
- cuDNN 10.1

```powershell
Measure-Command { waifu2x-ncnn-vulkan.exe -i input.png -o output.png -n 2 -s 2 -t [block size] -m [model dir] }
```

```powershell
Measure-Command { waifu2x-caffe-cui.exe -t 0 --gpu 0 -b 1 -c [block size] -p cudnn --model_dir [model dir] -s 2 -n 2 -m noise_scale -i input.png -o output.png }
```

### cunet

||Image Size|Target Size|Block Size|Total Time(s)|GPU Memory(MB)|
|---|---|---|---|---|---|
|waifu2x-ncnn-vulkan|200x200|400x400|400/200/100|0.86/0.86/0.82|638/638/197|
|waifu2x-caffe-cui|200x200|400x400|400/200/100|2.54/2.39/2.36|3017/936/843|
|waifu2x-ncnn-vulkan|400x400|800x800|400/200/100|1.17/1.04/1.02|2430/638/197|
|waifu2x-caffe-cui|400x400|800x800|400/200/100|2.91/2.43/2.7|3202/1389/1178|
|waifu2x-ncnn-vulkan|1000x1000|2000x2000|400/200/100|2.35/2.26/2.46|2430/638/197|
|waifu2x-caffe-cui|1000x1000|2000x2000|400/200/100|4.04/3.79/4.35|3258/1582/1175|
|waifu2x-ncnn-vulkan|2000x2000|4000x4000|400/200/100|6.46/6.59/7.49|2430/686/213|
|waifu2x-caffe-cui|2000x2000|4000x4000|400/200/100|7.01/7.54/10.11|3258/1499/1200|
|waifu2x-ncnn-vulkan|4000x4000|8000x8000|400/200/100|22.78/23.78/27.61|2448/654/213|
|waifu2x-caffe-cui|4000x4000|8000x8000|400/200/100|18.45/21.85/31.82|3325/1652/1236|

### upconv_7_anime_style_art_rgb

||Image Size|Target Size|Block Size|Total Time(s)|GPU Memory(MB)|
|---|---|---|---|---|---|
|waifu2x-ncnn-vulkan|200x200|400x400|400/200/100|0.74/0.75/0.72|482/482/142|
|waifu2x-caffe-cui|200x200|400x400|400/200/100|2.04/1.99/1.99|995/546/459|
|waifu2x-ncnn-vulkan|400x400|800x800|400/200/100|0.95/0.83/0.81|1762/482/142|
|waifu2x-caffe-cui|400x400|800x800|400/200/100|2.08/2.12/2.11|995/546/459|
|waifu2x-ncnn-vulkan|1000x1000|2000x2000|400/200/100|1.52/1.41/1.44|1778/482/142|
|waifu2x-caffe-cui|1000x1000|2000x2000|400/200/100|2.72/2.60/2.68|1015/570/459|
|waifu2x-ncnn-vulkan|2000x2000|4000x4000|400/200/100|3.45/3.42/3.63|1778/482/142|
|waifu2x-caffe-cui|2000x2000|4000x4000|400/200/100|3.90/4.01/4.35|1015/521/462|
|waifu2x-ncnn-vulkan|4000x4000|8000x8000|400/200/100|11.16/11.29/12.07|1796/498/158|
|waifu2x-caffe-cui|4000x4000|8000x8000|400/200/100|9.24/9.81/11.16|995/546/436|

## Sample Images

### Original Image

![origin](images/0.jpg)

### Upscale 2x with ImageMagick

```shell
convert origin.jpg -resize 200% output.png
```

![browser](images/1.png)

### Upscale 2x with ImageMagick Lanczo4 Filter

```shell
convert origin.jpg -filter Lanczos -resize 200% output.png
```

![browser](images/4.png)

### Upscale 2x with waifu2x noise=2 scale=2

```shell
waifu2x-ncnn-vulkan.exe -i origin.jpg -o output.png -n 2 -s 2
```

![waifu2x](images/2.png)

## Original waifu2x Project

- https://github.com/nagadomi/waifu2x
- https://github.com/lltcggie/waifu2x-caffe

## Other Open-Source Code Used

- https://github.com/Tencent/ncnn for fast neural network inference on ALL PLATFORMS
- https://github.com/webmproject/libwebp for encoding and decoding Webp images on ALL PLATFORMS
- https://github.com/nothings/stb for decoding and encoding image on Linux / MacOS
- https://github.com/tronkko/dirent for listing files in directory on Windows
