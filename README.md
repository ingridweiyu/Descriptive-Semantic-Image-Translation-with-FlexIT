# Descriptive Semantic Image Translation with FlexIT

This is the official implementation of Descriptive Semantic Image Translation with FlexIT.

## Setup

- Download `val2017.zip` from [COCO-Stuff](https://github.com/nightrome/cocostuff) and place the unzipped `val2017` directory into `data/cocostuff`.
- Download `stuff_trainval2017.zip` and `annotations_trainval2017.zip` from [COCO-Stuff](https://github.com/nightrome/cocostuff), unzip, and place all the JSON files into `data/cocostuff/annotations`.

## Natural language object detection using Stanford OpenIE:
- Run `code/openie/openie.py` with `python3 openie.py IMAGE_ID.csv "description of the transform (e.g. the red hat on the cat next to the winder is replaced with an umbrella)"`. It will output a CSV with the first row indicating the subject (target object) instances and the second row indicating the object (desired transformed object) instances.

## Evaluation with LPIPS and SFID
- Download `Evaluation_LPIPS_SFID.ipynb` and the `pytorch_sfid` directory and follow the instructions in the notebook.
