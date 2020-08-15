# PSPnet-Pytorch
Pytorch implementation of PSPnet for head/face segmentation(dataset is not allowed to provide according to the authors, but you can ask for a private download [link](https://www.mut1ny.com/face-headsegmentation-dataset))

# Summary
This is Pytorch version of PSPNet(adapted from [link](https://github.com/Lextal/pspnet-pytorch)). Support of pytorch 1.4.0.

# Dataset
![image](https://github.com/chuzcjoe/PSPnet-Pytorch/raw/master/img/seg.PNG)

# Details

1. Original implementation uses offical resnet weights, however the resent structure has been modified in the latest pytorch code.(FIXED)
2. Add dataloader scripts to read color images, convert segmentation image to label image and generate classification label.(FIXED)
![image](https://github.com/chuzcjoe/PSPnet-Pytorch/raw/master/img/label.PNG)
3. Add data augmentation includes: scale, flip and rotation. (TESTING)
4. Add single image testing. (FIXED)
