import os
import subprocess
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.utils import make_grid
from einops.einops import repeat, rearrange

from stylegan2.utils import noise, image_noise, styles_def_to_tensor, evaluate_in_chunks, slerp
from stylegan2.network import StyleGAN2
from stylegan2.srgan import Generator

class StyleGAN2Model():
    def __init__(self, model_path=None, models_dir='.models', results_dir='.results', upsample=False, **config):

        self.image_size = config.pop('image_size', 128)
        self.latent_dim = config.pop('latent_dim', 512)
        self.batch_size = config.pop('batch_size', 3)
        self.fmap_max = config.pop('fmap_max', 512)
        self.style_depth = config.pop('style_depth', 8)
        self.network_capacity = config.pop('network_capacity', 16)
        self.transparent = config.pop('transparent', False)
        self.attn_layers = config.pop('attn_layers', [1])
        self.steps = config.pop('steps', 1)
        self.lr_mlp = config.pop('lr_mlp', 0.1)
        self.av = None
        self.upsample = upsample

        # path to save results
        self.base_dir = Path(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

        self.results_dir = self.base_dir / results_dir
        (self.results_dir).mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # GAN model
        self.GAN = StyleGAN2(
            lr_mlp = self.lr_mlp,
            image_size = self.image_size,
            network_capacity = self.network_capacity,
            fmap_max = self.fmap_max,
            transparent = self.transparent,
            attn_layers = self.attn_layers,
            device = self.device
        )

        self.models_dir = self.base_dir / models_dir
        (self.models_dir).mkdir(parents=True, exist_ok=True)

        urls = {"StyleGAN2" : "https://www.dropbox.com/s/g5ap1n549qoudzj/model_45.pt?dl=0",
                "SRGAN_G" : "https://www.dropbox.com/s/x9ks2hnc7rswexo/best_g.pth.tar?dl=0"}

        if model_path is None:
            model_name = os.path.split(urls["StyleGAN2"])[-1][:-5]
            model_path = os.path.join(self.models_dir, model_name)

            # download pretrained model
            if not os.path.exists(model_path):
                subprocess.call(['wget', urls["StyleGAN2"], '-O', model_path])

        # load model
        print("Loading model from ...", model_path)
        load_data = torch.load(model_path, map_location = self.device)
        self.GAN.load_state_dict(load_data["GAN"])
        self.num_params = sum(p.numel() for p in self.GAN.parameters())

        # load things for sampling
        self.gen_n_layers = self.GAN.G.num_layers
        self.stylevec = self.GAN.S

        # automatically upsample if image size less than 256
        if self.image_size < 256:
            self.upsample = True

        # upsample images
        if self.upsample:
            # SRGAN generator
            self.srgan_gen = Generator(scale_factor=4)

            sr_gen_name = os.path.split(urls["SRGAN_G"])[-1][:-5]
            sr_gen_path = os.path.join(self.models_dir, sr_gen_name)

            # download pretrained SRGAN generator model
            if not os.path.exists(sr_gen_path):
                subprocess.call(['wget', urls["SRGAN_G"], '-O', sr_gen_path])

            # load SRGAN generator
            srgan_checkpoint = torch.load(sr_gen_path, map_location=self.device)
            self.srgan_gen.load_state_dict(srgan_checkpoint["state_dict"])
            self.srgan_gen.to(self.device)

    # truncates w in w-space so that values lie close to the mean
    @torch.no_grad()
    def truncate_style(self, tensor, trunc_psi = 0.75):
        S = self.GAN.S
        batch_size = self.batch_size
        latent_dim = self.GAN.G.latent_dim

        if self.av is None:
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
        w_truncated = self.truncate_style_defs(w, trunc_psi = trunc_psi) # [(B, latent_dim), num_layers]
        w_styles = styles_def_to_tensor(w_truncated) # [B, num_layers, latent_dim]
        generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi)
        return generated_images.clamp_(0., 1.)

    # generates images from interpolation
    @torch.no_grad()
    def generate_interpolation(self, name = 'sample', ratios = None, num_steps = 10, save_frames = False, sync_audio = False, audio_path = None):
        self.GAN.eval()

        if sync_audio:
            save_frames = True

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise
        latents_low = noise(1, latent_dim, device=self.device)  # [1, latent_dim]
        latents_high = noise(1, latent_dim, device=self.device) # [1, latent_dim]
        n = image_noise(1, image_size, device=self.device)      # [1, im_size, im_size, 1]

        # ratios for SLERP
        if ratios is None:
            ratios = torch.linspace(0., 8., num_steps)

        # generates images from interpolated latents
        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high) # [1, latent_dim]
            latents = [(interp_latents, num_layers)]
            generated_images = self.generate_truncated(self.GAN.S, self.GAN.G, latents, n, trunc_psi = 0.6)

            if self.upsample:
                hr_images = self.srgan_gen(generated_images)
            else:
                hr_images = generated_images

            pil_image = transforms.ToPILImage()(torch.squeeze(hr_images).cpu())

            if self.transparent:
                background = Image.new("RGBA", pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)

            frames.append(pil_image)
        frames[0].save(str(self.results_dir / f'{str(name)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

        # saves frames
        if save_frames:
            # path to save images
            folder_path = (self.results_dir / "frames" / f'{str(name)}')
            folder_path.mkdir(parents=True, exist_ok=True)

            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind).zfill(5)}.jpg'))

        # sync audio to generated images
        if sync_audio:
            # check for audio
            if audio_path is None:
                raise("No audio provided. Please provide audio.")

            video_path = str(self.results_dir/f'{str(name)}.mp4')

            # combine image sequences and audio into video
            subprocess.call(['ffmpeg',
                    '-r', '24',
                    '-i', f'{folder_path}/%05d.jpg',
                    '-i', audio_path,
                    '-r', '24',
                    '-vcodec', 'libx264',
                    '-y', video_path])

            return video_path

    # generate images from small uniform changes in latent space
    @torch.no_grad()
    def generate_from_latent(self, name, noise_z, noise):
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
        folder_path = (self.results_dir / f'{str(name)}')
        folder_path.mkdir(parents=True, exist_ok=True)

        # saves images
        for i, image in enumerate(tqdm(images)):
            pil_image = transforms.ToPILImage()(image.cpu())
            pil_image.save(str(folder_path / f'{str(i)}.jpg'))


    @torch.no_grad()
    def get_image_from_latents(self, z, noise):
        """get images from given input latent space (z) and noise.

        Args:
            z (np.ndarray or torch.Tensor): latent space vector with shape [b, latent_dim]
            noise (np.ndarray or torch.Tensor): base image size with shape [b, image_size, image_size, 1]

        Returns:
            np.ndarray: output images with shape [b, image_size, image_size, 3]
        """
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()
        if isinstance(noise, np.ndarray):
            noise = torch.from_numpy(noise).float()

        assert isinstance(z, torch.Tensor)
        assert isinstance(noise, torch.Tensor)

        styles = self.stylevec(z.to(self.device)) # [n, latent_dim]
        styles = repeat(styles, "b c -> b n c", n = self.gen_n_layers)
        out = self.GAN.G(styles = styles, input_noise = noise)
        out.clamp_(0., 1.)
        out = out.permute((0, 2, 3, 1)).cpu().numpy()

        return out

    @torch.no_grad()
    def sample(self, n, same_z = False, same_noise = False):
        """randomly sample `n` images.

        Args:
            n (int): number of images to generate
            same_z (bool): use same latent space for all images
            same_noise (bool): use same image noise for all images

        Returns:
            np.ndarray: output images with shape [n, image_size, image_size, 3]
        """
        if same_z and same_noise and n > 1:
            raise RuntimeError("same noise and same z will generate same images")

        # randomly create a z and noise vector and pass through the generator to get an image
        idim = self.image_size
        if same_z:
            z = torch.randn(self.latent_dim)
            z = repeat(z, "c -> n c", n = n)
        else:
            z = torch.randn(n, self.latent_dim)

        if same_noise:
            input_noise_image = torch.randn(idim, idim, 1)
            input_noise_image = repeat(input_noise_image, "h w c -> n h w c", n = n)
        else:
            input_noise_image = torch.randn(n, idim, idim, 1)

        out = self.get_image_from_latents(z, input_noise_image)

        return out


# ----- functions

def image_grid(x, n, as_uint8):
  if not isinstance(x, np.ndarray):
    x = np.array(x)
  x = torch.from_numpy(x)
  x = rearrange(x, "b h w c -> b c h w")
  g = make_grid(x, n)
  g = rearrange(g, "c h w -> h w c")
  g = g.numpy()
  if as_uint8:
    g = (g * 255).astype(np.uint8)
  return g
