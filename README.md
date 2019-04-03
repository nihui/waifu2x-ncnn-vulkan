# waifu2x-ncnn-vulkan
waifu2x converter ncnn version, runs fast on intel / amd / nvidia with vulkan

# the original waifu2x project
https://github.com/nagadomi/waifu2x

# download windows exe for your intel/amd/nvidia GPU card
https://github.com/nihui/waifu2x-ncnn-vulkan/releases

# Usage
```
waifu2x.exe [input image] [output png] [noise=-1/0/1/2/3] [scale=1/2]
```
* noise = noise level, large value means strong denoise effect, -1=no effect
* scale = scale level, 1=no scale, 2=upscale 2x

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

