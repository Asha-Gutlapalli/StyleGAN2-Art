# fire is a python library that automatically generates Command Line Interfaces (CLIs) from any python component
import fire
from tqdm import tqdm

import torch

from stylegan2.utils import cast_list, audio_features, timestamped_filename
from stylegan2.train import Trainer


# main function
def train_from_folder(
    data = './Trippy_Image_Dataset',
    results_dir = './.results',
    models_dir = './.models',
    audio_dir = './sample.wav',
    name = 'trippy',
    new = False,
    load_from = -1,
    image_size = 512,
    network_capacity = 16,
    fmap_max = 512,
    transparent = False,
    attn_layers = 1,
    batch_size = 3,
    gradient_accumulate_every = 40,
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
    num_image_tiles = 8,
    trunc_psi = 0.75,
    mixed_prob = 0.9,
    no_pl_reg = True,
    aug_prob = 0.3,
    aug_types = ['translation', 'cutout', 'color'],
    dataset_aug_prob = 0.6,
    calculate_fid_every = None,
    calculate_fid_num_images = 12800,
    clear_fid_cache = False,
    sync_audio = False,
    generate_latent = False
):
    # model arguments
    model_args = dict(
        name = name,
        results_dir = results_dir,
        models_dir = models_dir,
        batch_size = batch_size,
        gradient_accumulate_every = gradient_accumulate_every,
        image_size = image_size,
        network_capacity = network_capacity,
        fmap_max = fmap_max,
        transparent = transparent,
        attn_layers = attn_layers,
        lr = learning_rate,
        lr_mlp = lr_mlp,
        ttur_mult = ttur_mult,
        num_workers = num_workers,
        save_every = save_every,
        evaluate_every = evaluate_every,
        num_image_tiles = num_image_tiles,
        trunc_psi = trunc_psi,
        no_pl_reg = no_pl_reg,
        aug_prob = aug_prob,
        aug_types = cast_list(aug_types),
        dataset_aug_prob = dataset_aug_prob,
        calculate_fid_every = calculate_fid_every,
        calculate_fid_num_images = calculate_fid_num_images,
        clear_fid_cache = clear_fid_cache,
        mixed_prob = mixed_prob
    )

    # generates sample images
    if generate:
        model = Trainer(**model_args)
        model.load(load_from)
        samples_name = timestamped_filename()
        for num in tqdm(range(num_generate)):
            model.evaluate(f'{samples_name}-{num}', num_image_tiles)
        print(f'sample images generated at {results_dir}/{name}/{samples_name}')
        return

    # generates images from interpolation
    if generate_interpolation:
        if sync_audio:
            track = audio_features(audio_dir)
        else:
            track = None

        model = Trainer(**model_args)
        model.load(load_from)
        samples_name = timestamped_filename()
        model.generate_interpolation(samples_name, num_image_tiles, ratios = track, num_steps = interpolation_num_steps, save_frames = save_frames, sync_audio = sync_audio)
        print(f'interpolation generated at {results_dir}/{name}/{samples_name}')
        return

    # generate images from small changes in latent space
    if generate_latent:
        n = 10
        latent_dim = 512

        noise_z = torch.randn(1, latent_dim).repeat(n, 1)
        noise = torch.FloatTensor(n, image_size, image_size, 1).uniform_(0., 1.)

        model = Trainer(**model_args)
        model.load(load_from)
        samples_name = timestamped_filename()

        model.generate_latent(samples_name, noise_z, noise)

        print(f'sample images after small changes in latent space are generated at {results_dir}/{name}/{samples_name}')

        return

    model = Trainer(**model_args)

    # loads model from previous checkpoint unless specified otherwise
    if not new:
        model.load(load_from)
    else:
        model.clear()

    # loads dataset
    model.set_data_src(data)

    # trains model and prints log periodically
    for _ in tqdm(range(num_train_steps - model.steps), initial = model.steps, total = num_train_steps, mininterval=10., desc=f'{name}<{data}>'):
        model.train()
        if _ % 50 == 0:
            model.print_log()

    # saves model checkpoint
    model.save(model.checkpoint_num)

if __name__ == '__main__':
  # Fire exposes the contents of the program to the command line
  fire.Fire(train_from_folder)