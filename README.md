# StyleGAN2-Art [WIP]

StyleGAN is a Generative Adversarial Network proposed by NVIDIA researchers. It builds upon the [Progressive Growing GAN](https://arxiv.org/pdf/1710.10196.pdf)architecture to produce photo-realistic synthetic images. They have released three research papers regarding this as of June 2021. Their contributions are briefly summarized as follows:

- [StyleGAN](https://arxiv.org/pdf/1812.04948.pdf) [[Architecture](/diags/StyleGAN-Architecture.png)]:
  - **Bilinear Sampling:** to improve image quality.
  - **Mapping Network:** transforms latent space to w-space (Intermediate latent space).
  - **Synthesis Network:** uses [ADA-IN (Adaptive Instance Normalization)](https://arxiv.org/pdf/1703.06868.pdf) to generate images.
  - **Constant Initial Input:** increases performance. W-space and ADA-IN control generated images anyway.
  - **Gaussian Noise:** makes the generated image look more realistic by bringing in finer features.
  - **Mixing Regularization:** performs style mixing where images are generated from two intermediate styles.
  - **Perceptual Path Length:** measures the difference between successive images when interpolating between two noise inputs.
  - **Linear Separability:** separates the latent space with a linear hyperplane based on attribute.

- [StyleGAN2](https://arxiv.org/pdf/1912.04958.pdf) [[Architecture](/diags/StyleGAN2-Architecture.png)]:
  - **Weight Demodulation:** removes the droplet artifacts that were in the original StyleGAN.
  - **Lazy Regularization:** alleviates heavy memory usage and computation cost of regularization.
  - **Perceptual Path Length Regularization:** encourages smooth mapping from latent space to generated image to achieve feature disentanglement.
  - **No Growing:** replaces Progressive Growing GAN architecture to prevent phase artifacts with skip connections in the generator and residual connections in the discriminator.
  - **Large Networks:** yield better results where high-resolution images have more influence.

- [StyleGAN2-ADA](https://arxiv.org/pdf/2006.06676.pdf) [[Architecture](/diags/StyleGAN2-ADA-Architecture.png)]:
  - **Adaptive Discrimator Augmentation:** augments the data given to the discriminator to overcome overfitting without the augmentations leaking into generated images.


## Credits

This repository is adapted from Phil Wang's [`lucidrains/stylegan2-pytorch`](https://github.com/lucidrains/stylegan2-pytorch). I found his version of the NVlabs official PyTorch implementation of [`NVlabs/stylegan2-ada-pytorch`](https://github.com/NVlabs/stylegan2-ada-pytorch) to be easier to comprehend. His experiments are also interesting and informative.


## Getting Started

### Install libsndfile

Install "libsndfile" using the following commands for handling audio files in Linux:
```bash
sudo apt update -y
sudo apt install libsndfile1 -y
```

### Install ffmpeg

Install "ffmpeg" using the following commands:
```bash
sudo apt install ffmpeg [Ubuntu]
brew install ffmpeg     [macOS]
```

**Full install**
<details>

```
cd ~/ffmpeg_sources && \
wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
tar xjvf ffmpeg-snapshot.tar.bz2 && \
cd ffmpeg && \
PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
  --prefix="$HOME/ffmpeg_build" \
  --pkg-config-flags="--static" \
  --extra-cflags="-I$HOME/ffmpeg_build/include" \
  --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
  --extra-libs="-lpthread -lm" \
  --ld="g++" \
  --bindir="$HOME/bin" \
  --enable-gpl \
  --enable-gnutls \
  --enable-libaom \
  --enable-libass \
  --enable-libfdk-aac \
  --enable-libfreetype \
  --enable-libmp3lame \
  --enable-libopus \
  --enable-libsvtav1 \
  --enable-libdav1d \
  --enable-libvorbis \
  --enable-libvpx \
  --enable-libx264 \
  --enable-libx265 \
  --enable-nonfree && \
PATH="$HOME/bin:$PATH" make && \
make install && \
hash -r
```

</details>

### Install Packages

Install required libraries using the following command:
```bash
$ pip install -r requirements.txt
```


## Files

- `train.py`: Main file to kick start training, generate samples images, and generate images from interpolation.
- `stylegan2/`: All things StyleGAN2!
- `Architecture Diagrams/`: This folder consists of all the architecture diagrams from all three papers.
- `Study.md`: Study material
- `Examples/`: A folder of videos examples


## Train

Start training with the following command:
```bash
$ python train.py --data='/path/to/dataset'
```
There are various [arguments](./stylegan2/README.md#`train_from_folder`) that can be used as command line inputs to this command.


## Inference

### Generate Sample Images

Generate sample images from the latest checkpoint after training with the following command:
```bash
$ python train.py --generate
```


### Generate Images/Video from Interpolation

Generate a video of interpolation from the latest checkpoint after training with the following command:
```bash
$ python train.py --generate-interpolation
```

Generate video and images from interpolation from the latest checkpoint after training with the following command:
```bash
$ python train.py --generate-interpolation --save-frames
```


## Run Streamlit App

Use the following command to run Streamlit App:
```bash
$ streamlit run run.py

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.4:8501

```

Check out the GIF below for demo!

<img src="app.gif">
