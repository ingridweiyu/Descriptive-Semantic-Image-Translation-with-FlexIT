import os

os.chdir("FlexIT")
from demo import *
os.chdir("..")

import mdetr
import os
from PIL import Image


# The following dictionary contains only select images we ran.
# These images are the ones we included in our report.
images = {"000000000632": ("a basket on the right", "a bucket"),
          "000000003501": ("the broccoli on the left", "lettuce"),
          "000000006954": ("the frisbee on the right", "a dinner plate"),
          "000000012667": ("the banana", "a cucumber")
         }

for image_no in images:
    print("Translating image " + image_no)
    
    image = image_no + ".png"
    source = images[image_no][0]
    target = images[image_no][1]
    
    image_path = "data_png/" + image
    bb = tuple(mdetr.get_bb(image_path, source))
    square_len = bb[2] - bb[0]
    
    orig = Image.open(image_path)
    crop = orig.crop(bb)
    
    out_img = translate(img_path = image_path,
                        pil_img = crop,
                        S = source,
                        T = target,
                        save_dir = "translated")
    out_img = out_img.resize((square_len, square_len))
    orig.paste(out_img, (bb[0], bb[1]))
    orig.save("translated/"+image)