# FCN-MultiBox Detector

FMD ( Full convolution MultiBox Detector ) is a simplified SSD ( https://github.com/weiliu89/caffe/tree/ssd ) with enabling different input sizes.
It is just a learning project to master all details of a detector. Code is impllemented in Torch, is very simple and easy to understand.


## 1. Network

* VGG16 convoluting image (224x224, or other sizes) to feature layer (14x14x512), just a typical full convolution network. 

* Following feature layer, apply several kernels with different size and shape, such as 2x2, 2x3, 3x3, 2x4 ...  

* Every kernel outputs a class score or a binding box according it's position (in feature layer) and kernel size. 
  These kernels are just sliding windows with different postion and different size.

* Like SSD, not every predicted box do BP, just some selected ones, (positve and negtive is 1:4).

See model.lua, and boxsampleing.lua

## 2. Training 

### 2.1 prepare code

In data folder, just see list.sh, filter.py, and makedb.lua. Combing all the info of VOC into a single t7 file.

### 2.2 training 

See makefixed.lua, use fixed 10 layers from VGG16. Then see train.lua, it is very simple. 
The data batch is build in different threads ( see data_loader.lua and data_process.lua).

## 3. Detecting

See doDetect.lua. The following is result of random pictures from web.

<p>
<img src="/demo.jpg" height="360px" style="max-width:100%;">
</p>

