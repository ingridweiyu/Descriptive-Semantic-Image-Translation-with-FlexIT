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

To obtain images for testing translation, please visit one of the following Google Drive links.
- Small version (117.9 MB, 250 images): [Google Drive]()
- Large version (2.4 GB, 5000 images): [Google Drive]()

### FlexIT

FlexIT requires setup before it can function properly. After cloning our GitHub repository, unzip `FlexIT.zip`, and enter the commands:
```
pip install -r requirements.txt
bash install.sh
```
When running the code, additional `ModuleNotFoundError`s may occur. Simply download the necessary package as indicated by each error message.

The VQGAN component of FlexIT relies on two files which FlexIT incorrectly obtains. This will cause two error messages, respectively related to files called `model.yaml` and `last.ckpt`. You'll need to download these two files manually from:
- https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1
- https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1
Place these two files in the cache location as indicated by the above referenced error message.

Unfortunately, these problems are beyond our control, as FlexIT is one component of our final architecture that we did not implement ourselves. For more information on diverse usages of FlexIT, visit their [GitHub repository](https://github.com/facebookresearch/SemanticImageTranslation).


# Run Code
From `root`, enter the following command into Terminal.
```
python main.py
```

# Credits

