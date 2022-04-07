#------------------------------------------------------------------------------
# This section code is borrowed from https://github.com/CompVis/taming-transformers
# under scripts/taming-transformers.ipynb and
# scripts/reconstruction_usage.ipynb.

# ! git clone https://github.com/CompVis/taming-transformers

import numpy as np
from omegaconf import OmegaConf
import os
import PIL
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import yaml

os.chdir('taming-transformers')
from taming.models.vqgan import VQModel, GumbelVQ
os.chdir('..')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    if is_gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()


def preprocess_image(img, target_image_size=256):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
    
    img = TF.resize(img, (target_image_size, target_image_size), interpolation=PIL.Image.LANCZOS)
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return img


def preprocess_vqgan(x):
    x = 2.*x - 1.
    return x


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x
#------------------------------------------------------------------------------


def load(config_path, ckpt_path):
    '''
    Loads VQGAN model
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    ckpt_path : str
        Path to model checkpoint file
        
    Returns
    -------
    model : taming.models.cond_transformer.Net2NetTransformer
        Pre-trained VQGAN conditional transformer model
    
    '''
    os.chdir('taming-transformers')
    
    if config_path[0:20] == 'taming-transformers/':
        config_path = config_path[20:]
        
    if ckpt_path[0:20] == 'taming-transformers/':
        ckpt_path = ckpt_path[20:]
    
    config = load_config(config_path, display=False)
    model = load_vqgan(config, ckpt_path=ckpt_path).to(device)
    
    os.chdir('..')
    return model


def prep_image_for_vqgan(img, size = 256):
    '''
    Opens an image and preprocesses it to prepare for VQGAN encoding

    Parameters
    ----------
    img_path : PIL.Image.Image
        Image
    size : int
        Target image size

    Returns
    -------
    None.

    '''
    image = preprocess_image(img, size)
    image = preprocess_vqgan(image)
    return image


def revert_size(img, orig_size):
    '''
    Rescale an image to i

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    orig_size : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    