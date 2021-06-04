# StyleGAN2-Art [WIP]

StyleGAN is a Generative Adversarial Network proposed by NVIDIA researchers. It builds upon the <a href="https://arxiv.org/pdf/1710.10196.pdf">Progressive Growing GAN</a> architecture to produce photo-realistic synthetic images. They have released three research papers regarding this as of June 2021. Their contributions are briefly summarized as follows:

- <a href="https://arxiv.org/pdf/1812.04948.pdf">StyleGAN </a><a href="./Architecture Diagrams/StyleGAN-Architecture.png">[Architecture]</a>:
  - <b>Bilinear Sampling</b> to improve image quality.
  - <b>Mapping Network</b> transforms latent space to w-space (Intermediate latent space).
  - <b>Synthesis Network</b> uses <a href="https://arxiv.org/pdf/1703.06868.pdf">ADA-IN (Adaptive Instance Normalization)</a> to generate images.
  - <b>Constant Initial Input</b> increases performance. W-space and ADA-IN control generated images anyway.
  - <b>Gaussian Noise</b> makes the generated image look more realistic by bringing in finer features.
  - <b>Mixing Regularization</b> performs style mixing where images are generated from two intermediate styles.
  - <b>Perceptual Path Length</b> measures the difference between successive images when interpolating between two noise inputs.
  - <b>Linear Separability</b> separates the latent space with a linear hyperplane based on attribute.

- <a href="https://arxiv.org/pdf/1912.04958.pdf">StyleGAN2 </a><a href="./Architecture Diagrams/StyleGAN2-Architecture.png">[Architecture]</a>:
  - <b>Weight Demodulation</b> removes the droplet artifacts that were in the original StyleGAN.
  - <b>Lazy Regularization</b> alleviates heavy memory usage and computation cost of regularization.
  - <b>Perceptual Path Length Regularization</b> encourages smooth mapping from latent space to generated image to achieve feature disentanglement.
  - <b>No Growing</b> replaces Progressive Growing GAN architecture to prevent phase artifacts with skip connections in the generator and residual connections in the discriminator.
  - <b>Large Networks</b> yield better results where high-resolution images have more influence.

- <a href="https://arxiv.org/pdf/2006.06676.pdf">StyleGAN2-ADA </a><a href="./Architecture Diagrams/StyleGAN2-ADA-Architecture.png">[Architecture]</a>:
  - <b>Adaptive Discrimator Augmentation</b> augments the data given to the discriminator to overcome overfitting without the augmentations leaking into generated images.


## Credits

This repository is adapted from Phil Wang's <a href="https://github.com/lucidrains/stylegan2-pytorch">"Simple StyleGan2 for PyTorch"</a>. I found his version of the NVlabs official PyTorch implementation of <a href="https://github.com/NVlabs/stylegan2-ada-pytorch">"StyleGAN2-ADA"</a> to be easier to comprehend. His experiments are also interesting and informative.


## Install Packages

Install required libraries using the following command:
```bash
$ pip install -r requirements.txt
```


## Files

- `train.py`: Main file to kick start training, generate samples images, and generate images from interpolation.
- `stylegan2/`: All things StyleGAN2!
- `Architecture Diagrams/`: This folder consists of all the architecture diagrams from all three papers.
- `Study.md`: Study material


## Train

Start training with the following command:
```bash
$ python train.py --data='/path/to/dataset'
```
There are various [arguments](./stylegan2/README.md#`train_from_folder`) that can be used as command line inputs to this command.


## Generate Sample Images

Generate sample images from the latest checkpoint after training with the following command:
```bash
$ python train.py --generate
```


## Generate Images/Video from Interpolation

Generate a video of interpolation from the latest checkpoint after training with the following command:
```bash
$ python train.py --generate-interpolation
```

Generate video and images from interpolation from the latest checkpoint after training with the following command:
```bash
$ python train.py --generate-interpolation --save-frames
```