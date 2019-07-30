# waifu2x ncnn Vulkan

ncnn implementation of waifu2x converter. Runs fast on Intel / AMD / Nvidia with Vulkan API.

waifu2x-ncnn-vulkan uses [ncnn project](https://github.com/Tencent/ncnn) as the universal neural network inference framework.

## [Download](https://github.com/nihui/waifu2x-ncnn-vulkan/releases)

Download Windows Executable for Intel/AMD/Nvidia GPU

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
  -i input-path        input image path (jpg/png) or directory
  -o output-path       output image path (png) or directory
  -n noise-level       denoise level (-1/0/1/2/3, default=0)
  -s scale             upscale ratio (1/2, default=2)
  -t tile-size         tile size (>=32, default=400)
  -m model-path        waifu2x model path (default=models-cunet)
  -g gpu-id            gpu device to use (default=0)
  -j load:proc:save    thread count for load/proc/save (default=1:2:2)
```

- `input-path` and `output-path` accept either file path or directory path
- `noise-level` = noise level, large value means strong denoise effect, -1=no effect
- `scale` = scale level, 1=no scale, 2=upscale 2x
- `tile-size` = tile size, use smaller value to reduce GPU memory usage, default is 400
- `load:proc:save` = thread count for the three stages (image decoding + waifu2x upscaling + image encoding), use larger value may increase GPU utility and consume more GPU memory. You can tune this configuration as "4:4:4" for many small-size images, and "2:2:2" for large-size images. The default setting usually works fine for most situations. If you find that your GPU is hungry, do increase thread count to achieve faster processing.

If you encounter crash or error, try to upgrade your GPU driver

- Intel: https://downloadcenter.intel.com/product/80939/Graphics-Drivers
- AMD: https://www.amd.com/en/support
- NVIDIA: https://www.nvidia.com/Download/index.aspx

## Speed Comparison with waifu2x-caffe-cui

### Environment

- Windows 10 1809
- AMD R7-1700
- Nvidia GTX-1070
- Nvidia driver 419.67
- CUDA 10.1.105
- cuDNN 10.1

```powershell
Measure-Command { waifu2x-ncnn-vulkan.exe -i input.png -o output.png -n 2 -s 2 [block size] -m [model dir] }
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

## ncnn Project (>=20190712)

- https://github.com/Tencent/ncnn/tree/8c537069875a28d5380c6bdcbf7964d73803b7b3

## Other Open-Source Code Used

- https://github.com/nothings/stb for decoding and encoding image on Linux / MacOS
- https://github.com/tronkko/dirent for listing files in directory on Windows
