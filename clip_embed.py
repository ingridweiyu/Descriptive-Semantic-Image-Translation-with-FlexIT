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



model, preprocess = clip.load(model_name)

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
    
    bb = coco.im2label[image_id][0][0]
    xmin, ymin, width, height = bb[0], bb[1], bb[2], bb[3]
    xmax = xmin + width
    ymax = ymin + height
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    
    filename = ''
    for i in range(12 - len(str(image_id))):
        filename += '0'
    filename += str(image_id) + '.jpg'
    
    image_path = os.path.join(image_dir, filename)
    image = Image.open(image_path)
    cropped = image.crop((xmin, ymin, xmax, ymax))
    
    cropped = preprocess(cropped).unsqueeze(0)
    sourcetext = clip.tokenize([S])
    targettext = clip.tokenize([T])

    # CLIP embedding of the cropped bear image
    image_term = model.encode_image(cropped)
    # CLIP embedding of "a bear"
    sourcetext_term = model.encode_text(sourcetext)
    # CLIP embedding of "an elephant"
    targettext_term = model.encode_text(targettext)

    # Multimodal target point P from FlexIT
    P = lambda_I * image_term + targettext_term - lambda_S * sourcetext_term
    
    return P



# Example:
P_bear_elephant = P('285', 'data/cocostuff/val2017', 'a bear standing on the grass', 'a tall elephant walking by')
print(P_bear_elephant)