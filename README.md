# Descriptive Semantic Image Translation with FlexIT

This is the official implementation of Descriptive Semantic Image Translation with FlexIT.

## Directory Structure

```
root
│   README.md
|   main.py
|   losses.py
|   clip_embed.py
|   coco.py
│   get_coco_im2label.py
|   vqgan_encdec.py
|   Citations.txt
|   Evaluation_LPIPS_SFID.ipynb
└───code
    └───class_id_openie
        |   openie.py
        |   openie_sample-output.csv
    └───eval
        └───pytorch_sfid
            |   __init__.py
            |   __main__.py
            |   inception.py
            |   sfid_score.py
└───data
    └───cocostuff
        |   val2017_im2label.json
```

## Setup

### Data

- Download [COCO-Stuff] module, and create directories in Google Drive
- Download `val2017.zip` from [COCO-Stuff](https://github.com/nightrome/cocostuff) and place the unzipped `val2017` directory into `data/cocostuff`.
- Download `stuff_trainval2017.zip` and `annotations_trainval2017.zip` from [COCO-Stuff](https://github.com/nightrome/cocostuff), unzip, and place all the JSON files into `data/cocostuff/annotations`.
- Download `Evaluation_LPIPS_SFID.ipynb` and follow instructions in the notebook to install. Make sure to clone `PerceptualSimilarity` into `root`.

### VQGAN

- In `root`, run `git clone https://github.com/CompVis/taming-transformers`. 
- Download [pre-trained VQGAN](https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1) and the corresponding [configuration file](https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1).
   - Place the pre-trained VQGAN `last.ckpt` into `taming-transformers/logs/vqgan_imagenet_f16_1024/checkpoints` and create these directories.
   - Place the corresponding configuration file `model.yaml` into `taming-transformers/logs/vqgan_imagenet_f16_1024/configs`.

## Natural language object detection using Stanford OpenIE:
- Run `code/openie/openie.py` with `python3 openie.py IMAGE_ID.csv "description of the transform (e.g. the red hat on the cat next to the winder is replaced with an umbrella)"`. It will output a CSV with the first row indicating the subject (target object) instances and the second row indicating the object (desired transformed object) instances.

## Evaluation with LPIPS and SFID
- Download the `pytorch_sfid` directory and follow the instructions in the notebook. For SFID, regularization term alpha is set to 1.

# Run Main Code
From `root`, enter the following command into Terminal.
```
python main.py
```
