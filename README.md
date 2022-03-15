# Digital-rock-image-processing
This repository is for codes and models used in the following paper:
[Reconstructing high fidelity digital rock images using deep convolutional neural networks](https://www.nature.com/articles/s41598-022-08170-8)

## Description 

* Codes include models for image denoising, deblurring, and single-image super-resolution 
* Models were developed and trained using TensorFlow version 2.4
* Pre-trained models are not added due to size limit at this time  

## Models:

* Denoiser network is based on the model by [Yu et al. (2019)](https://ieeexplore.ieee.org/document/9025411) 
* Deblurring network is based on the model by [Cho et al. (2021)](https://arxiv.org/abs/2108.05054)
* Super-resolution networks:
    * DFCAN based on [Qiao et al. (2021)](https://www.nature.com/articles/s41592-020-01048-5)
    * EDSR based on [Lim et al. (2017)](https://arxiv.org/abs/1707.02921) 
    * WDSR based on [Yu et al. (2018)](https://arxiv.org/abs/1808.08718)
* Codes for EDSR and WDSR partly from [Martin Krasser](https://github.com/krasserm/super-resolution.git) work

## Data 
* Training and test data for this work were prepared from publicly avaiable data from [Digitial Rock portal](https://www.digitalrocksportal.org/)
