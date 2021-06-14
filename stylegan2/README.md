# StyleGAN2

## Files

- `utils.py`: Helper classes and functions
- `augment.py`: Augmentation functions and wrapper class for ADA
- `data.py`: Custom dataset class and helper functions
- `network.py`: StyleGAN2 network
- `model.py`: Simplified StyleGAN2 model


## [`StyleGAN2`](./stylegan2/network.py)

```python
class StyleGAN2(nn.Module):
    def __init__(
      self,
      image_size,            #  Image size
      latent_dim = 512,      # Latent dimension
      fmap_max = 512,        # Maximum filter size
      style_depth = 8,       # Number of layers in Mapping Network
      network_capacity = 16, # Network capacity
      transparent = False,   # True if RGBA images and False otherwise
      attn_layers = [],      #  Number of attention layers
      steps = 1,             # Number of steps
      lr = 1e-4,             # Learning rate
      ttur_mult = 2,         # Learning rate multiplier for Adam optimizer
      lr_mlp = 0.1,          # Learning rate multiplier for MLP(Multilayer Perceptron)
      device = "cuda"        # CPU or GPU
    )
```


## [`train_from_folder`](./train.py)

You can either refer the python script below or use the following command for more information.
```bash
$ python train.py --help
```

```python
def train_from_folder(
    data = './data',                                # image dataset
    results_dir = './.results',                     # Path to save results
    models_dir = './.models',                       # Path to save models
    audio_dir = './sample.wav',                     # Path to audio file
    name = 'trippy',                                # project name
    new = False,                                    # flag to training a new model, else from previous model
    load_from = -1,                                 # Checkpoint number or -1 if loads model from last checkpoint
    image_size = 512,                               # Side of image dimension
    network_capacity = 16,                          # Capacity of Network
    fmap_max = 512,                                 # Maximum filter size
    transparent = False,                            # True for RGBA images and False otherwise
    attn_layers = 1,                                # Number of attention layers
    batch_size = 3,                                 # Batch size
    gradient_accumulate_every = 40,                 # Number of gradient accumulations
    num_train_steps = 150000,                       # Number of training steps or epochs
    learning_rate = 2e-4,                           # Learning rate
    lr_mlp = 0.1,                                   # Learning rate multiplier for MLP(Multilayer Perceptron)
    ttur_mult = 1.5,                                # Learning rate multiplier for Adam optimizer
    num_workers =  None,                            # Number of workers
    save_every = 1000,                              # Once in number of steps to save model
    evaluate_every = 1000,                          # Once in number of steps to evaluate and save results
    generate = False,                               # Whether or not to generate sample images
    num_generate = 1,                               # Number of sample images to be generated
    generate_interpolation = False,                 # Whether or not generate images from interpolation
    interpolation_num_steps = 100,                  # Number of interpolation steps
    save_frames = False,                            # Whether or not to save frames
    num_image_tiles = 8,                            # Number of image tiles on each side
    trunc_psi = 0.75,                               # Truncation value
    mixed_prob = 0.9,                               # Probability for mixed noise
    no_pl_reg = True,                               # Whether or not to calculate Perceptual Path Length Regularization
    aug_prob = 0.3,                                 # Augmentation probability for ADA
    aug_types = ['translation', 'cutout', 'color'], # Augmentation types for ADA
    dataset_aug_prob = 0.6,                         # Augmentation probability for dataset
    calculate_fid_every = None,                     # Once in number of steps to calculate and save FID score
    calculate_fid_num_images = 12800,               # Number of images for which FID is calculated
    clear_fid_cache = False,                        # Clears FID cache
    sync_audio = False,                             # Whether or not to sync audio to generated images
    generate_latent = False                         # Whether or not to generate images after small uniform changes in latent space
)
```
