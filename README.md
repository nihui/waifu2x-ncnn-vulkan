# waifu2x-ncnn-vulkan
waifu2x converter ncnn version, runs fast on intel / amd / nvidia with vulkan

waifu2x-ncnn-vulkan use ncnn project [https://github.com/Tencent/ncnn] as the universal nerual network inference framework

# download windows exe for your intel/amd/nvidia GPU card
https://github.com/nihui/waifu2x-ncnn-vulkan/releases

This package includes all the binary and models required, it is portable, no cuda or caffe runtime needed :)

# usage
```
waifu2x.exe [input image] [output png] [noise=-1/0/1/2/3] [scale=1/2] [blocksize=400]
```
* noise = noise level, large value means strong denoise effect, -1=no effect
* scale = scale level, 1=no scale, 2=upscale 2x
* blocksize = tile size, use smaller value to reduce GPU memory usage, default is 400

If you encounter crash or error, try to upgrade your GPU driver

intel https://downloadcenter.intel.com/product/80939/Graphics-Drivers

amd https://www.amd.com/en/support

nvidia https://www.nvidia.com/Download/index.aspx

# speed compared with waifu2x-caffe-cui

Windows 10 1809, AMD R7-1700, Nvidia GTX-1070, Nvidia driver 419.67, CUDA 10.1.105, cudnn 10.1

```
Measure-Command { waifu2x.exe input.png output.png 2 2 [block size] }
```
```
Measure-Command { waifu2x-caffe-cui.exe -t 0 --gpu 0 -b 1 -c [block size] -p cudnn --model_dir ./models/cunet -s 2 -n 2 -m noise_scale -o input.png -i output.png }
```

||image size|target size|block size|total time(s)|GPU memory(MB)|
|---|---|---|---|---|---|
|waifu2x-ncnn-vulkan|200x200|400x400|400/200/100|0.62/0.62/0.61|332/332/96|
|waifu2x-caffe-cui|200x200|400x400|400/200/100|2.54/2.39/2.36|3017/936/843|
|waifu2x-ncnn-vulkan|400x400|800x800|400/200/100|0.99/0.92/0.94|1226/348/96|
|waifu2x-caffe-cui|400x400|800x800|400/200/100|2.91/2.63/2.7|3202/1389/1178|
|waifu2x-ncnn-vulkan|1000x1000|2000x2000|400/200/100|3.05/3.03/3.29|1245/348/97|
|waifu2x-caffe-cui|1000x1000|2000x2000|400/200/100|4.04/3.79/4.35|3258/1582/1175|
|waifu2x-ncnn-vulkan|2000x2000|4000x4000|400/200/100|10.16/10.53/11.61|1264/367/97|
|waifu2x-caffe-cui|2000x2000|4000x4000|400/200/100|7.01/7.54/10.11|3258/1499/1200|
|waifu2x-ncnn-vulkan|4000x4000|8000x8000|400/200/100|38.73/40.26/44.87|1468/385/115|
|waifu2x-caffe-cui|4000x4000|8000x8000|400/200/100|18.45/21.85/31.82|3325/1652/1236|

# sample
## original image
![origin](https://raw.githubusercontent.com/nihui/waifu2x-ncnn-vulkan/master/0.jpg)
## upscale 2x with ImageMagick
```
convert origin.jpg -resize 200% output.png
```
![browser](https://raw.githubusercontent.com/nihui/waifu2x-ncnn-vulkan/master/1.png)
## upscale 2x with ImageMagick Lanczo4 filter
```
convert origin.jpg -filter Lanczos -resize 200% output.png
```
![browser](https://raw.githubusercontent.com/nihui/waifu2x-ncnn-vulkan/master/4.png)
## upscale 2x with waifu2x noise=2 scale=2
```
waifu2x.exe origin.jpg output.png 2 2
```
![waifu2x](https://raw.githubusercontent.com/nihui/waifu2x-ncnn-vulkan/master/2.png)

# original waifu2x project
https://github.com/nagadomi/waifu2x

https://github.com/lltcggie/waifu2x-caffe

# ncnn project (>=20190413)
https://github.com/Tencent/ncnn/tree/c180e875026b6f6ffa0f6e3e489a5c1487439b7f
