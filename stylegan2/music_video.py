import os
import math
import librosa
import numpy as np
from PIL import Image
import librosa.display
from tqdm import trange

from stylegan2.model import image_grid


def get_chromagram(data, sr):
  """get chromagram for given audio

  Args:
      data (np.ndarray): 1D float numpy array
      sr (int): sampling rate

  Returns:
      np.ndarray: 2D chromagram
  """
  # this is the complete process of getting the beats-chroma
  tempo, beat_frames = librosa.beat.beat_track(y=data, sr=sr)
  y_harmonic, y_percussive = librosa.effects.hpss(data)
  chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length = 512)
  beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)
  # plt.imshow(beat_chroma)
  return chromagram

def get_latents_from_sound(data, sr, chromagram, z_dim = 512):
  """This function is the heart of this project, it converts the input
  audio to z-latents that can be fed directly into the model.

  Args:
      data (np.ndarray): 1D float numpy array
      sr (int): sampling rate
      chromagram (np.ndarray): 2D float numpy array

  Returns:
      np.ndarray: 2D latents
  """
  # define a few numbers for refenrence
  fps = 30
  nsec = len(data) // sr
  total_steps = int(nsec * fps)
  split_size = math.ceil(len(data) / total_steps)

  # we need to get the chromagram at the end of every second
  # NOTE: split is done for +1 seconds since we need to get the transition for
  # the last second as well.
  ratio = chromagram.shape[1] / (nsec + 1)
  chroma_per_second = chromagram[:, [int(round(i * ratio)) for i in range(int(nsec) + 1)]]
  mat = np.random.uniform(size = (chromagram.shape[0], z_dim - 64))
  red = chroma_per_second.T @ mat / 10
  red += 64 # shift values
  # plt.hist(v_synced_chroma.reshape(-1))
  # plt.hist(red.reshape(-1), )

  red = red.argmax(-1)
  # print(red, red.shape)

  # Z - latent: at each second you will get a decrease in the value of current spike idx and
  # rise at next spike idx. Now there are a couple of different ways the value can reduce:
  # a) sigmoid function (used here)
  # b) linear decrease
  z_full = []
  sec_ctr = 0
  _sp = np.linspace(-5, 5, fps)
  incr_sigmoidal_transition = 1 / (1 + np.exp(-_sp))
  decr_sigmoidal_transition = 1 / (1 + np.exp(+_sp))
  for i in range(total_steps):
    if i % fps == 0:
      s = red[sec_ctr]; e = red[sec_ctr+1]
      z = np.zeros(512,)
      z[s] = 1
      # ln = np.linspace(s, e, fps)
      sec_ctr += 1
    else:
      _i = i % fps
      z = np.zeros(512,)
      if s == e:
        # special case where the frequency actually remains the same and you start
        # getting patterns like: [... 0.98674903, 0.99057736, 0.99330715,
        # 1.        , 0.00942264, 0.01325097, 0.0186055 ...] with the signoidal thing below
        
        # one thing to do is to keep the value 1 all the way till the end of second
        # if there is a change, well and good. else it will continue to remain the same
        # pattern
        z[s] = 1.
      else:
        z[s] = decr_sigmoidal_transition[_i]
        z[e] = incr_sigmoidal_transition[_i]
    z_full.append(z)
  z_full = np.array(z_full)

  # plt.figure(figsize = (15, 10))
  # plt.imshow(z_full.T)
  # print(z_full.shape)

  return z_full

def get_images_from_latents(model, z_full, fps = 30):
  """once the latents are generated you can now use them
  to build images

  Args:
      model (StyleGAN2Model): model to be used
      z_full (np.ndarray): 2D float array
      fps (int, optional): frames per second. Defaults to 30.

  Returns:
      list: list of PIL.Image
  """
  all_out = []
  for i in trange(fps * 30):
    b = 4
    z = np.repeat(z_full[i].reshape(1, 512), [b], axis = 0)
    z[np.arange(b), np.arange(b)] = 1.

    noise = np.ones((b, model.image_size, model.image_size, 1))
    samples = model.get_image_from_latents(z, noise)
    all_out.append(Image.fromarray(image_grid(samples, 4, True)))
  
  return all_out

def save_image(images, path):
  for _i,i in zip(trange(len(images)), images):
    i.save(os.path.join(path, f"img{_i}.png"))
