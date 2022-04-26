# Descriptive Semantic Image Translation with FlexIT

This is the official implementation of Descriptive Semantic Image Translation with FlexIT.

## Setup

### Data

To obtain images for testing translation, please visit one of the following Google Drive links. The images in the large version contains all images in the small version.
- Small version (117.9 MB, 250 images): [Google Drive](https://drive.google.com/drive/folders/1vO0P1uS0ylmLfytLZeonX5YG7MmyKd1k?usp=sharing)
- Large version (2.4 GB, 5000 images): [Google Drive](https://drive.google.com/drive/folders/1WbaB7ev09Z7pLM4ZKcZjd9WFNMPMQG8R?usp=sharing)

Place the downloaded directory under this repository's root directory and rename the folder to `data_png`.

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

And, place these two files in the cache location as indicated by the above referenced error message.

We apologize for this inconvenience. Unfortunately, these problems are beyond our control, as FlexIT is one component of our final architecture that we did not implement ourselves. For more information on diverse usages of FlexIT, visit their [GitHub repository](https://github.com/facebookresearch/SemanticImageTranslation).


## Run Code

### Image Editing
To edit images, modify the dictionary called `images` in `dsit.py`. The keys are filenames without their extensions, and values are tuples of soure text and target text. Clear, illustrative examples are presently available in the file. Run the program using:
```
python dsit.py
```
Translated images will be saved to `translated`.

### Evaluation
`dsit-eval.py` computes an average LPIPS similarity score between translated images and original images. Please make sure all original and translated files are available in their respective directories.
```
python dsit-eval.py
```

## Credits

- The contents of `FlexIT.zip` were obtained directly from [FlexIT's original repository](https://github.com/facebookresearch/SemanticImageTranslation). However, we made slight modifications to resolve errors and wrapped certain code in callable functions.
- `PerceptualSimilarity` was obtained directly from [here](https://github.com/richzhang/PerceptualSimilarity).
- `taming` was automatically generated from code that we did not write.
- `mdetr.py` was derived from [MDETR's Python notebook](https://colab.research.google.com/github/ashkamath/mdetr/blob/colab/notebooks/MDETR_demo.ipynb). We adapted their code only slightly.

There is [more code](https://drive.google.com/file/d/1M5fKv621kRHFX5BoEHG9IYH_tHwsxLz1/view?usp=sharing) we wrote as part of this project's process which are not included in this repository. We wrote this to manually implement FlexIT, before we adopted their official implementation. Running them as-is will not work without complex dependencies, but we wanted to make them available to demonstrate our process.
