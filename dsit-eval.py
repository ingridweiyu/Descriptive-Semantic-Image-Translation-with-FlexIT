import os
from PIL import Image
import torch
import torchvision.transforms as T

os.chdir('PerceptualSimilarity')
import lpips
os.chdir('..')



lpips_vgg = lpips.LPIPS(net='vgg')

original = os.listdir("data_png")
translated = os.listdir("translated")
    

def L_perc(img1, img2):
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
    img1_tensor = torch.unsqueeze(T.ToTensor()(img1), 0)
    img2_tensor = torch.unsqueeze(T.ToTensor()(img2), 0)
    return lpips_vgg(img1_tensor, img2_tensor)


num_images = len(translated)
count = 0

lpips_sum = 0
for image in translated:
    count += 1
    print("Processing image", count, "/", num_images)
    orig_img = Image.open("data_png/"+image)
    trans_img = Image.open("translated/"+image)
    lpips_sum += L_perc(orig_img, trans_img)

lpips_avg = lpips_sum / len(translated)
print("AVerage LPIPS score:", lpips_avg)