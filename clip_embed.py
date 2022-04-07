import clip
import cv2
import get_coco_im2label as coco
import os
from PIL import Image
import sys
import torch


# ! pip install ftfy regex tqdm
# ! pip install git+https://github.com/openai/CLIP.git


lambda_I = 0.2
lambda_S = 0.4
model_name = 'RN50'



clip_model, clip_preprocess = clip.load(model_name)



def image_filename(image_id):
    '''
    Generates the image filename corresponding to image_id, according to the
    format used by cocostuff/val2017

    Parameters
    ----------
    image_id : str
        Image ID number

    Returns
    -------
    filename : str
        Filename corresponding to image_id
    '''
    filename = ''
    for i in range(12 - len(str(image_id))):
        filename += '0'
    filename += str(image_id) + '.jpg'
    
    return filename



def I0(image_id, image_dir, bb_index = 0):
    '''
    Crops an image down to a specified bounding box

    Parameters
    ----------
    image_id : str
        Image ID number
    image_dir : str
        Path to directory containing image. Current directory is represented
        by the empty string.
    bb_index : int
        Index of which bounding box corresponding to the image to use

    Returns
    -------
    cropped : PIL.Image.Image
        Image cropped according to specified bounding box

    '''
    filename = image_filename(image_id)
    image_path = os.path.join(image_dir, filename)
    image = Image.open(image_path)
    
    bb = coco.im2label[image_id][bb_index][0]
    xmin, ymin, width, height = bb[0], bb[1], bb[2], bb[3]
    xmax = xmin + width
    ymax = ymin + height
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    
    cropped = image.crop((xmin, ymin, xmax, ymax))
    return cropped


def image_to_clip(image):
    '''
    Encodes an image to the CLIP multimodal space

    Parameters
    ----------
    image : PIL.Image.Image
        Image to be embedded by CLIP

    Returns
    -------
    image_encoded : torch.Tensor
        Embedding of the image in the CLIP multimodal space
    '''
    im = clip_preprocess(image).unsqueeze(0)
    im_encoded = clip_model.encode_image(im)
    return im_encoded



def image_bb_to_clip(image_id, image_dir):
    '''
    Encodes an image to the CLIP multimodal space according to its bounding box
    
    Parameters
    ----------
    image_id : str
        Image ID number
    image_dir : str
        Path to directory containing image. Current directory is represented
        by the empty string.
    
    Returns
    -------
    image_encoded : torch.Tensor
        Embedding of the image in the CLIP multimodal space
    '''
    image = I0(image_id, image_dir, 0)
    image_encoded = image_to_clip(image)
    return image_encoded


def text_to_clip(text):
    '''
    Encodes a text to the CLIP multimodal space

    Parameters
    ----------
    text : str
        Text to be encoded

    Returns
    -------
    text_encoded : torch.Tensor
        Embedding of the text in the CLIP multimodal space
    '''
    tokenized = clip.tokenize([text])
    text_encoded = clip_model.encode_text(tokenized)
    return text_encoded


def P(image_id, image_dir, S, T):
    '''
    Computes the multimodal target point P
    
    Parameters
    ----------
    image_id : str
        Image ID number
    image_dir : str
        Path to directory containing image. Current directory is represented
        by the empty string.
    S : str
        Source text
    T : str
        Target text
    
    Returns
    -------
    P : torch.Tensor
        Embedding of target point in the CLIP multimodal space
    '''

    image_term = image_bb_to_clip(image_id, image_dir)
    sourcetext_term = text_to_clip(S)
    targettext_term = text_to_clip(T)

    # Multimodal target point P from FlexIT
    P = lambda_I * image_term + targettext_term - lambda_S * sourcetext_term
    
    return P


