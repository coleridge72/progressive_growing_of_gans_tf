# Progressive Growing of GANs for Improved Quality, Stability, and Variation on TensorFlow

This repository contains the **TensorFlow re-implementation (both training and sampling)** of [original implementation (Theano and Lasagne)](https://github.com/tkarras/progressive_growing_of_gans). The author of this repo is actively training and debugging on 1024x1024 CelebA dataset.

**Please checkout our [image inpainting](https://github.com/JiahuiYu/generative_inpainting) project as well. [Demo](http://jhyu.me/demo) is available for high-resolution CelebA-HQ face inpainting!**

## Dataset: 1024x1024 CelebA faces

PNG images can be downloaded from [Google Drive](https://drive.google.com/open?id=1_BsFYYGVvcjcPNhPam2STBDFG3eKAdKr) (download_7z).

<img src="https://user-images.githubusercontent.com/22609465/33633338-0b825560-d9d6-11e7-8177-2840f9dd9a93.png" width="425"/> <img src="https://user-images.githubusercontent.com/22609465/33633711-5aa770a2-d9d7-11e7-9517-916c8169d984.png" width="425"/> 
Example High-Resolution CelebA images

## Training

The author of this repo is actively training and debuging on 1024x1024 CelebA dataset. If you want to try early, please modify `progressive_gan.yml` accordingly and run `python train.py`.

## TensorBoard

Visualization on TensorBoard is supported.

## License

License inherits from original [NVIDIA License](https://github.com/tkarras/progressive_growing_of_gans/blob/master/LICENSE).

## Requirements

* [NeuralGym 0.0.1-alpha](https://github.com/JiahuiYu/neuralgym)
* [TensorFlow >= 1.4](https://www.tensorflow.org/)
