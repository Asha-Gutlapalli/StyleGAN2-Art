import torch

from stylegan2.utils import audio_features, noise, timestamped_filename
from stylegan2.model import StyleGAN2Model

# file paths
audio_path = './sample.wav'
model_path = '/home/asha/StyleGAN2-Art/.models/trippy/model_45.pt'

# get ratios from audio features for interpolation
ratios = audio_features(file_path = audio_path)

# load model
model = StyleGAN2Model(model_path = model_path)

# generate images and sync audio
model.generate_interpolation(name = timestamped_filename(), ratios = ratios, sync_audio = True, audio_path = audio_path)

#noise
n = 10
latent_dim = 512
image_size = 128

noise_z = torch.randn(1, latent_dim).repeat(n, 1)
noise = torch.FloatTensor(n, image_size, image_size, 1).uniform_(0., 1.)

# generate images from uniform changes in latent space
model.generate_latent(name = timestamped_filename(), noise_z = noise_z, noise = noise)
