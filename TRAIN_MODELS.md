## HOW TO TRAIN YOUR OWN MODELS
/!\ This is a quick draft from what i understood how things works. it's not meant to be pushed in production and it was not fully tested (like not at all in fact). Be aware of that thank you very much /!\

How to train your own model and convert them for waifu2x-ncnn-vulkan.

Here's a quick guide to help people training their own models. I took the time to understand how things worked and why everyone used the same dataset but in differents formats.
So you'll have to convert a waifu2x model to a ncnn model, this is the only way to get it working on waifu2x-ncnn-vulkan.


#### waifu2x
https://github.com/nagadomi/waifu2x

To train your models you'll have to follow waufu2x's training method. 
Once you finished training, you can use waifu2x's tools to convert dataset to a `.json` file and get a `.prototxt` file.

#### waifu2x-caffe
https://github.com/lltcggie/waifu2x-caffe

```*.caffemodel is automatically generated from *.prototxt and *.json when waifu2x-caffe is launched. When replacing json files, delete *.caffemodel before launching waifu2x-caffe.```
Once you have your `.json` and `.prototxt` you convert as a `.caffemodel` by just launching waifu2x-caffe.

#### ncnn
https://github.com/Tencent/ncnn

Finally you just have to use ncnn to convert `caffemodel` to a `.bin` and `.param` files.


With theses two files you can use ncnn or waifu2x-ncnn-vulkan.
