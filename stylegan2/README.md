# StyleGAN2

## Files

- `utils.py`: Helper classes and functions
- `augment.py`: Augmentation functions and wrapper class for ADA
- `data.py`: Custom dataset class and helper functions
- `network.py`: StyleGAN2 network


## `StyleGAN2` [./stylegan2/network.py]

```python
class StyleGAN2(nn.Module):
    def __init__(
      self,
      image_size,
      latent_dim = 512,
      fmap_max = 512,
      style_depth = 8,
      network_capacity = 16,
      transparent = False,
      attn_layers = [],
      steps = 1,
      lr = 1e-4,
      ttur_mult = 2,
      lr_mlp = 0.1,
      device = "cuda"):

    '''
    Args:
      image_size (int): Image size
      latent_dim (int, optional): Latent dimension
      fmap_max (int, optional): Maximum filter size
      style_depth (int, optional): Number of layers in Mapping Network
      network_capacity (int, optional): Network capacity
      transparent (bool, optional): True if RGBA images and False otherwise
      attn_layers (list(int), optional): Number of attention layers
      steps (int): Number of steps
      lr (float, optional): Learning rate
      ttur_mult (float, optional): Learning rate multiplier for Adam optimizer
      lr_mlp (float, optional): Learning rate multiplier for MLP(Multilayer Perceptron)
      device (str): CPU or GPU
    '''
```


## `train_from_folder` [./train.py]

```python
def train_from_folder(
    data = './data',
    results_dir = './.cache_results',
    models_dir = './.cache_models',
    name = 'trippy',
    new = False,
    load_from = -1,
    image_size = 128,
    network_capacity = 16,
    fmap_max = 512,
    transparent = False,
    attn_layers = 1,
    batch_size = 3,
    gradient_accumulate_every = 10,
    num_train_steps = 150000,
    learning_rate = 2e-4,
    lr_mlp = 0.1,
    ttur_mult = 1.5,
    num_workers =  None,
    save_every = 1000,
    evaluate_every = 1000,
    generate = False,
    num_generate = 1,
    generate_interpolation = False,
    interpolation_num_steps = 100,
    save_frames = False,
    num_image_tiles = 2,
    trunc_psi = 0.75,
    mixed_prob = 0.9,
    no_pl_reg = True,
    aug_prob = 0.,
    aug_types = ['translation', 'cutout'],
    dataset_aug_prob = 0.,
    calculate_fid_every = None,
    calculate_fid_num_images = 12800,
    clear_fid_cache = False,
    sync_audio = False,
    audio_feature_type = "beats",
    change_z = False
):

'''
This is the main function to train and generate images!

Args:
  data (str): Path to image dataset
  results_dir (str, optional): Path to save results
  models_dir (str, optional): Path to save models
  name: Name of the project
  new (bool): Restarts training
  load_from (int, optional): Checkpoint number or -1 if loads model from last checkpoint
  image_size (int, optional): Image size
  network_capacity (int, optional): Capacity of Network
  fmap_max (int, optional): Maximum filter size
  transparent (bool, optional): True for RGBA images and False otherwise
  attn_layers (int or list(int), optional): Number of attention layers
  batch_size (int, optional): Batch size
  gradient_accumulate_every (int, optional): Number of gradient accumulations
  num_train_steps (int, optional): Number of training steps or epochs
  learning_rate (float, optional): Learning rate
  lr_mlp (float, optional): Learning rate multiplier for MLP(Multilayer Perceptron)
  ttur_mult (float, optional): Learning rate multiplier for Adam optimizer
  num_workers (int, optional): Number of workers
  save_every (int, optional): Once in number of steps to save model
  evaluate_every (int, optional): Once in number of steps to evaluate and save results
  generate (bool, optional): Whether or not to generate sample images
  num_generate (int, optional): Number of sample images to be generated
  generate_interpolation (bool, optional): Whether or not generate images from interpolation
  interpolation_num_steps (int, optional): Number of interpolation steps
  save_frames (bool, optional): Whether or not to save frames
  num_image_tiles (int, optional): Number of image tiles on each side
  trunc_psi (float, optional): Truncation value
  mixed_prob (float, optional): Probability for mixed noise
  no_pl_reg (bool, optional): Whether or not to calculate Perceptual Path Length Regularization
  aug_prob (float, optional): Augmentation probability for ADA
  aug_types (list(str), optional): Augmentation types for ADA
  dataset_aug_prob (float, optional): Augmentation probability for dataset
  calculate_fid_every (int, optional): Once in number of steps to calculate and save FID score
  calculate_fid_num_images (int, optional): Number of images for which FID is calculated
  clear_fid_cache (bool, optional): Clears FID cache
  sync_audio (bool, optional): Whether or not to sync audio to generated images
  audio_feature_type (str, optional): Type of audio feature
  change_z (bool, optional): Whether or not to generate images after small uniform changes in latent space
'''
```