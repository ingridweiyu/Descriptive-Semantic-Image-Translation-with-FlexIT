import clip_embed as c
from get_coco_im2label import *
import losses
import torch
import torchvision.transforms.functional as TF
import vqgan_encdec as vq


image_id = '285'
image_dir = 'data/cocostuff/val2017'

I0 = c.I0(image_id, image_dir) # PIL.Image.Image
I0_size = (I0.size[1], I0.size[0])

P_bear_elephant = c.P(image_id, image_dir, 'a bear standing on the grass', 'a tall elephant walking by')


vqgan_model = vq.load('taming-transformers/logs/vqgan_imagenet_f16_1024/configs/model.yaml',
                      'taming-transformers/logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt')


# Example image to pass through VQGAN
image = vq.prep_image_for_vqgan(I0)

# Result of image encoder to get z0
z0, _, [_, _, indices] = vqgan_model.encode(image)
z = torch.rand(z0.shape, requires_grad = True)


# AT THIS POINT IN CODE, MANIPULATE z0 TO TRANSLATE TOWARDS TARGET.

lr = 0.5

for i in range(150):
    decoded = vqgan_model.decode(z)
    decoded_image = vq.custom_to_pil(decoded[0])
    decoded_image = TF.resize(decoded_image, I0_size)
    
    print('Iteration', i)
    if i % 5 == 0:
        filename = "iter" + str(i) + ".jpg"
        decoded_image.save("output/"+filename)
    
    L_emb = losses.L_emb(decoded_image, P_bear_elephant)
    L_perc = losses.L_perc(decoded_image, I0)
    L_latent = losses.L_latent(z0, z)
    L_total = losses.L_total(L_emb, L_perc, L_latent)
    
    L_total.backward(inputs = z)
    with torch.no_grad():
        z -= lr * z.grad
        z.grad = None