# waifu2x-ncnn-vulkan
waifu2x converter ncnn version, runs fast on intel / amd / nvidia with vulkan

waifu2x-ncnn-vulkan use ncnn project [https://github.com/Tencent/ncnn] as the universal nerual network inference framework

# download windows exe for your intel/amd/nvidia GPU card
https://github.com/nihui/waifu2x-ncnn-vulkan/releases

This package includes all the binary and models required, it is portable, no cuda or caffe runtime needed :)

# Usage
```
waifu2x.exe [input image] [output png] [noise=-1/0/1/2/3] [scale=1/2]
```
* noise = noise level, large value means strong denoise effect, -1=no effect
* scale = scale level, 1=no scale, 2=upscale 2x

If you encounter crash or error, try to upgrade your GPU driver

intel https://downloadcenter.intel.com/product/80939/Graphics-Drivers

amd https://www.amd.com/en/support

nvidia https://www.nvidia.com/Download/index.aspx

# Sample
## original image
![origin](https://raw.githubusercontent.com/nihui/waifu2x-ncnn-vulkan/master/0.jpg)
## upscale 2x with browser
![browser](https://raw.githubusercontent.com/nihui/waifu2x-ncnn-vulkan/master/1.png)
## upscale 2x with waifu2x noise=2 scale=2
```
waifu2x.exe origin.jpg output.png 2 2
```
![waifu2x](https://raw.githubusercontent.com/nihui/waifu2x-ncnn-vulkan/master/2.png)

# original waifu2x project
https://github.com/nagadomi/waifu2x

https://github.com/lltcggie/waifu2x-caffe

# ncnn project (>=20190406)
https://github.com/Tencent/ncnn/tree/c5ab0c86e4d8ee70c375d1cea49bb82a580e418c
