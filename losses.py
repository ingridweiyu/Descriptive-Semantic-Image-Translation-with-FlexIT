from clip_embed import *
import os
import torch
import torchvision.transforms as T

os.chdir('PerceptualSimilarity')
import lpips
os.chdir('..')

lpips_vgg = lpips.LPIPS(net='vgg')

def L_emb(image, P):
    '''
    Computes the embedding loss between an image and a target point P

    Parameters
    ----------
    image : PIL.Image.Image
        Image for calculating distance to P
    P : torch.Tensor
        Target point P in the CLIP multimodal space

    Returns
    -------
    loss : torch.Tensor
        Embedding loss between image and target point P
    '''
    image_clip = image_to_clip(image)
    loss = torch.sum((image_clip - P)**2)
    return loss


def L_perc(image, I0):
    '''
    Computes the perceptual loss between a translated image and its original
    version I0

    Parameters
    ----------
    image : PIL.Image.Image
        Translated image to calculate perceptual loss against original image
    I0 : PIL.Image.Image
        Original input image

    Returns
    -------
    None.
    '''
    image_tensor = torch.unsqueeze(T.ToTensor()(image), 0)
    I0_tensor = torch.unsqueeze(T.ToTensor()(I0), 0)
    return lpips_vgg(image_tensor, I0_tensor)