import json
import subprocess
import numpy as np
from tqdm import tqdm
from PIL import Image
from math import floor
from pathlib import Path
from shutil import rmtree

import torchvision
from torch.utils import data
from torchvision import transforms

from .network import *

# Training Code
class Trainer():
    def __init__(
        self,
        name = 'default',
        results_dir = 'results',
        models_dir = 'models',
        base_dir = './',
        image_size = 512,
        network_capacity = 16,
        fmap_max = 512,
        transparent = False,
        batch_size = 4,
        mixed_prob = 0.9,
        no_pl_reg = False,
        gradient_accumulate_every=1,
        lr = 2e-4,
        lr_mlp = 0.1,
        ttur_mult = 2,
        num_workers = None,
        save_every = 1000,
        evaluate_every = 1000,
        num_image_tiles = 8,
        trunc_psi = 0.6,
        attn_layers = [],
        aug_prob = 0.25,
        aug_types = ['translation', 'cutout', 'color'],
        dataset_aug_prob = 0.,
        calculate_fid_every = None,
        calculate_fid_num_images = 12800,
        clear_fid_cache = False,
        *args,
        **kwargs
    ):
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = base_dir / results_dir
        self.models_dir = base_dir / models_dir
        self.fid_dir = base_dir / 'fid' / name
        self.config_path = self.models_dir / name / '.config.json'

        assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.image_size = image_size
        self.network_capacity = network_capacity
        self.fmap_max = fmap_max
        self.transparent = transparent

        self.attn_layers = cast_list(attn_layers)

        self.aug_prob = aug_prob
        self.aug_types = aug_types

        self.lr = lr
        self.lr_mlp = lr_mlp
        self.ttur_mult = ttur_mult
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob

        self.num_image_tiles = num_image_tiles
        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        self.av = None
        self.trunc_psi = trunc_psi

        self.no_pl_reg = no_pl_reg
        self.pl_mean = None

        self.gradient_accumulate_every = gradient_accumulate_every

        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = None
        self.last_fid = None

        self.pl_length_ma = EMA(0.99)

        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = calculate_fid_every
        self.calculate_fid_num_images = calculate_fid_num_images
        self.clear_fid_cache = clear_fid_cache


    # Checks if images are transparent
    # JPG - Not transparent
    # PNG - Transparent
    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    # returns checkpoint number
    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)

    # initializes StyleGAN2 model
    def init_GAN(self):
        args, kwargs = self.GAN_params
        self.GAN = StyleGAN2(lr = self.lr, lr_mlp = self.lr_mlp, ttur_mult = self.ttur_mult, image_size = self.image_size, network_capacity = self.network_capacity, fmap_max = self.fmap_max, transparent = self.transparent, attn_layers = self.attn_layers, device = self.device, *args, **kwargs)

    # writes configuration to a config file
    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    # loads configuration from a config file
    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        self.transparent = config['transparent']
        self.attn_layers = config.pop('attn_layers', [])
        self.fmap_max = config.pop('fmap_max', 512)
        self.lr_mlp = config.pop('lr_mlp', 0.1)
        del self.GAN
        self.init_GAN()

    # returns configuration
    def config(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 'lr_mlp': self.lr_mlp, 'transparent': self.transparent, 'attn_layers': self.attn_layers,}

    # loads dataset
    def set_data_src(self, folder):
        self.dataset = data.Dataset(folder, self.image_size, transparent = self.transparent, aug_prob = self.dataset_aug_prob)
        num_workers = num_workers = default(self.num_workers, NUM_CORES)
        sampler = None
        dataloader = data.DataLoader(self.dataset, num_workers = num_workers, batch_size = self.batch_size, sampler = sampler, shuffle = False, drop_last = True, pin_memory = True)
        self.loader = cycle(dataloader)

        # auto set augmentation prob for user if dataset is detected to be low
        num_samples = len(self.dataset)
        if not exists(self.aug_prob) and num_samples < 1e5:
            self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
            print(f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')

    # trains model
    def train(self):

        # checks if dataloader exists
        assert exists(self.loader), 'You must first initialize the data source with `.set_data_src(<folder of images>)`'

        # initializes GAN if not already
        if not exists(self.GAN):
            self.init_GAN()

        self.GAN.train()

        # initializes total generator and discriminator loss
        total_disc_loss = torch.tensor(0.).to(self.device)
        total_gen_loss = torch.tensor(0.).to(self.device)

        batch_size = self.batch_size

        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers

        # setup augmentation arguments
        aug_prob   = self.aug_prob
        aug_types  = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}

        # applies gradient penalty periodically
        apply_gradient_penalty = self.steps % 4 == 0

        # applies perceptual path length penalty periodically
        apply_path_penalty = not self.no_pl_reg and self.steps > 5000 and self.steps % 32 == 0

        S = self.GAN.S
        G = self.GAN.G
        D = self.GAN.D
        D_aug = self.GAN.D_aug

        # setup losses
        D_loss_fn = hinge_loss
        G_loss_fn = gen_hinge_loss


        # train discriminator

        # sets gradients of discriminator's paramters to zero to clear old gradients
        self.GAN.D_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every):

            # get latents and noise
            get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
            style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.device)
            noise = image_noise(batch_size, image_size, device=self.device)

            # latent space to w space
            w_space = latent_to_w(S, style)
            w_styles = styles_def_to_tensor(w_space)

            # generated images
            generated_images = G(w_styles, noise)
            # output of generated images from dicriminator with augmentation
            fake_output = D_aug(generated_images.clone().detach(), detach = True, **aug_kwargs)

            # load next batch of images
            image_batch = next(self.loader).to(self.device)
            image_batch.requires_grad_()

            # output of real images from dicriminator with augmentation
            real_output = D_aug(image_batch, **aug_kwargs)

            # discriminator loss function
            divergence = D_loss_fn(real_output, fake_output)
            disc_loss = divergence

            # applies gradient penalty and calculates gradient penalty loss
            if apply_gradient_penalty:
                gp = gradient_penalty(image_batch, real_output)
                self.last_gp_loss = gp.clone().detach().item()
                disc_loss = disc_loss + gp

            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.register_hook(raise_if_nan)
            # propagates loss backwards
            loss_backwards(disc_loss)

            total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

        # discriminator loss
        self.d_loss = float(total_disc_loss)

        # performs parameter update
        self.GAN.D_opt.step()


        # train generator
        avg_pl_length = self.pl_mean
        self.GAN.G_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every):

            # get latents and noise
            style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.device)
            noise = image_noise(batch_size, image_size, device=self.device)

            # latent space to w space
            w_space = latent_to_w(S, style)
            w_styles = styles_def_to_tensor(w_space)

            # generated images
            generated_images = G(w_styles, noise)
            fake_output = D_aug(generated_images, **aug_kwargs)
            fake_output_loss = fake_output

            real_output = None

            # generator loss function
            loss = G_loss_fn(fake_output_loss, real_output)
            gen_loss = loss

            # applies perceptual path length penalty
            if apply_path_penalty:
                pl_lengths = calc_pl_lengths(w_styles, generated_images)
                avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

                # calculates perceptual path length loss
                if not is_empty(self.pl_mean):
                    pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                    if not torch.isnan(pl_loss):
                        gen_loss = gen_loss + pl_loss

            gen_loss = gen_loss / self.gradient_accumulate_every
            gen_loss.register_hook(raise_if_nan)
            # propagates loss backwards
            loss_backwards(gen_loss)

            total_gen_loss += loss.detach().item() / self.gradient_accumulate_every

        # generator loss
        self.g_loss = float(total_gen_loss)

        # performs parameter update
        self.GAN.G_opt.step()

        # calculates exponential moving average for perceptual path length
        if apply_path_penalty and not np.isnan(avg_pl_length):
            self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)

        # save from NaN errors
        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)

        # saves model checkpoint periodically
        if self.steps % self.save_every == 0:
            self.save(self.checkpoint_num)

        # saves intermediate results periodically
        if self.steps % 100 == 0:
            self.evaluate(floor(self.steps / self.evaluate_every))

        # calculates fid
        if exists(self.calculate_fid_every) and self.steps % self.calculate_fid_every == 0 and self.steps != 0:
            num_batches = math.ceil(self.calculate_fid_num_images / self.batch_size)
            fid = self.calculate_fid(num_batches)
            self.last_fid = fid

            with open(str(self.results_dir / self.name / f'fid_scores.txt'), 'a') as f:
                f.write(f'{self.steps},{fid}\n')

        self.steps += 1
        self.av = None

    # evaluate model
    @torch.no_grad()
    def evaluate(self, num = 0, trunc = 1.0):
        self.GAN.eval()

        ext = self.image_extension

        # image grid side dimension
        num_rows = self.num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise
        latents = noise_list(num_rows ** 2, num_layers, latent_dim, device=self.device)
        n = image_noise(num_rows ** 2, image_size, device=self.device)

        # generates images and save image grid
        generated_images = self.generate_truncated(self.GAN.S, self.GAN.G, latents, n, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)

    # calculate fid
    @torch.no_grad()
    def calculate_fid(self, num_batches):
        from pytorch_fid import fid_score
        torch.to(self.device).empty_cache()

        # setup paths to save fid scores for real and fake images
        real_path = self.fid_dir / 'real'
        fake_path = self.fid_dir / 'fake'

        # remove any existing files used for fid calculation and recreate directories
        if not real_path.exists() or self.clear_fid_cache:
            rmtree(real_path, ignore_errors=True)
            os.makedirs(real_path)

            # calculate fid for real images
            for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
                real_batch = next(self.loader)
                for k, image in enumerate(real_batch.unbind(0)):
                    filename = str(k + batch_num * self.batch_size)
                    torchvision.utils.save_image(image, str(real_path / f'{filename}.png'))

        # generate a bunch of fake images in results / name / fid_fake
        rmtree(fake_path, ignore_errors=True)
        os.makedirs(fake_path)

        self.GAN.eval()

        ext = self.image_extension

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # calculates fid for generated images
        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):

            # latents and noise
            latents = noise_list(self.batch_size, num_layers, latent_dim, device=self.device)
            noise = image_noise(self.batch_size, image_size, device=self.device)

            generated_images = self.generate_truncated(self.GAN.S, self.GAN.G, latents, noise, trunc_psi = self.trunc_psi)

            for j, image in enumerate(generated_images.unbind(0)):
                torchvision.utils.save_image(image, str(fake_path / f'{str(j + batch_num * self.batch_size)}.{ext}'))

        return fid_score.calculate_fid_given_paths([str(real_path), str(fake_path)], 256, noise.device, 2048)

    # truncates w in w-space so that values lie close to the mean
    @torch.no_grad()
    def truncate_style(self, tensor, trunc_psi = 0.75):
        S = self.GAN.S
        batch_size = self.batch_size
        latent_dim = self.GAN.G.latent_dim

        if not exists(self.av):
            z = noise(2000, latent_dim, device=self.device)
            samples = evaluate_in_chunks(batch_size, S, z).cpu().numpy()

            # average of w in w-space
            self.av = np.mean(samples, axis = 0)
            self.av = np.expand_dims(self.av, axis = 0)

        av_torch = torch.from_numpy(self.av).to(self.device)
        tensor = trunc_psi * (tensor - av_torch) + av_torch # [N, latent_dim]
        return tensor

    @torch.no_grad()
    def truncate_style_defs(self, w, trunc_psi = 0.75):
        w_space = []
        for tensor, num_layers in w:
            tensor = self.truncate_style(tensor, trunc_psi = trunc_psi)
            w_space.append((tensor, num_layers)) # [(N, latent_dim), num_layers]
        return w_space

    # generates images from truncated styles
    @torch.no_grad()
    def generate_truncated(self, S, G, style, noi, trunc_psi = 0.75, num_image_tiles = 8):
        w = map(lambda t: (S(t[0]), t[1]), style)
        w_truncated = self.truncate_style_defs(w, trunc_psi = trunc_psi) # [(N, latent_dim), num_layers]
        w_styles = styles_def_to_tensor(w_truncated) # [N, num_layers, latent_dim]
        generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi)
        return generated_images.clamp_(0., 1.)

    # generates images from interpolation
    @torch.no_grad()
    def generate_interpolation(self, num = 0, num_image_tiles = 8, trunc = 1.0, ratios = None, num_steps = 100, save_frames = False, sync_audio = False, audio_path = None):
        self.GAN.eval()

        ext = self.image_extension
        num_rows = num_image_tiles

        if sync_audio:
            save_frames = True

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise
        latents_low = noise(1, latent_dim, device=self.device)
        latents_high = noise(1, latent_dim, device=self.device)
        n = image_noise(1, image_size, device=self.device)

        # ratios for SLERP
        if ratios is None:
            ratios = torch.linspace(0., 8., num_steps)

        # generates images from interpolated latents
        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            latents = [(interp_latents, num_layers)]
            generated_images = self.generate_truncated(self.GAN.S, self.GAN.G, latents, n, trunc_psi = self.trunc_psi)
            images_grid = torchvision.utils.make_grid(generated_images, nrow = num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())

            if self.transparent:
                background = Image.new("RGBA", pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)

            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

        # saves frames
        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind).zfill(5)}.{ext}'))

        # sync audio to generated images
        if sync_audio:
            # check for audio
            if audio_path is None:
                raise("No audio provided. Please provide audio.")

            video_path = str(self.results_dir / self.name / f'{str(num)}.mp4')

            # combine image sequences and audio into video
            subprocess.call(['ffmpeg',
                    '-r', '24',
                    '-i', f'{folder_path}/%05d.jpg',
                    '-i', audio_path,
                    '-r', '24',
                    '-vcodec', 'libx264',
                    '-y', video_path])

    # generate images from small uniform changes in latent space
    @torch.no_grad()
    def generate_from_latent(self, num, noise_z, noise):
        ext = self.image_extension

        # noise
        z = noise_z.to(self.device)
        noise = noise.to(self.device)

        # uniformly distributed latent space values
        uniform_latent_values = torch.linspace(-1., 1., z.shape[0])

        # replaces original latent space values with uniformly distributed latent space values in one dimension
        for i, zi in enumerate(z):
            z[i][0] =  uniform_latent_values[i]

        num_layers = self.GAN.G.num_layers
        z_def = [(z, num_layers)]

        # generates images from latent space
        images = self.generate_truncated(self.GAN.S, self.GAN.G, z_def, noise, trunc_psi = 0.75)

        # path to save images
        folder_path = (self.results_dir / self.name / f'{str(num)}')
        folder_path.mkdir(parents=True, exist_ok=True)

        # saves images
        for i, image in enumerate(tqdm(images)):
            pil_image = transforms.ToPILImage()(image.cpu())
            pil_image.save(str(folder_path / f'{str(i)}.{ext}'))

    # prints out log
    def print_log(self):
        data = [
            ('G', self.g_loss), # generator loss
            ('D', self.d_loss), # discriminator loss
            ('GP', self.last_gp_loss), # gradient penalty loss
            ('PL', self.pl_mean), # perceptual path length loss
            ('FID', self.last_fid) # last fid score
        ]

        data = [d for d in data if exists(d[1])]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        print(log)

    # returns path to save model
    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    # creates folders for saving models and results if not already
    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    # removes existing folders for saving models, results, fid scores, and configurations
    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.fid_dir), True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    # saves model checkpoint and configuration
    def save(self, num):
        save_data = {
            'GAN': self.GAN.state_dict(),
        }

        torch.save(save_data, self.model_name(num))
        self.write_config()

    # loads model from model checkpoint
    def load(self, num = -1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name), map_location = self.device)

        try:
            self.GAN.load_state_dict(load_data['GAN'])
        except Exception as e:
            print('unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e
