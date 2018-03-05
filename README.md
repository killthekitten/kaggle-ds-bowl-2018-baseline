# kaggle-ds-bowl-2018-baseline

Full train/inference/submission pipeline adapted to the [Data Science Bowl competition](https://www.kaggle.com/c/data-science-bowl-2018/) from https://github.com/matterport/Mask_RCNN. Kudos to [@matterport](https://github.com/matterport), [@waleedka](https://github.com/waleedka) and others for the code. It is well written, but is also somewhat opinionated, which makes it harder to guess what's going on under the hood, which is the reason for my fork to exist.

I did almost no changes to the original code, except for:

* Everything custom in `bowl_config.py` and `bowl_dataset.py`.
* `VALIDATION_STEPS` and `STEPS_PER_EPOCH` are now forced to depend on the dataset size, hardcoded.
* `multiprocessing=False`, hardcoded.
* [@John1231983](https://github.com/John1231983)'s changes from [this PR](https://github.com/killthekitten/kaggle-ds-bowl-2018-baseline/pull/1).
* Added `RESNET_ARCHITECTURE` variable to the config (`resnet50` or `resnet101` while 101 comes with a default config).

## Quick Start

1. First, you have to download the train masks. Thanks [@lopuhin](https://github.com/lopuhin/) for bringing all the fixes to one place. You might want to do it outside of this repo to be able to pull changes later and use symlinks:

```bash
git clone https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes ../kaggle-dsbowl-2018-dataset-fixes
ln -s ../kaggle-dsbowl-2018-dataset-fixes/stage1_train stage1_train
```

2. Download the rest of the official dataset and unzip it to the repo:

```bash
unzip ~/Downloads/stage1_test.zip -d stage1_test
unzip ~/Downloads/stage1_train_labels.csv.zip -d .
unzip ~/Downloads/stage1_sample_submission.csv.zip -d .
```

3. Install `pycocotools` and COCO pretrained weights (`mask_rcnn_coco.h5`). General idea is described [here](https://github.com/matterport/Mask_RCNN#installation). Keep in mind, to install pycocotools properly, it's better to run `make install` instead of `make`.

4. For a single GPU training, run:

```
CUDA_VISIBLE_DEVICES="0" python train.py
```

5. To generate a submission, run:

```
CUDA_VISIBLE_DEVICES="0" python inference.py
```

This will create `submission.csv` in the repo and overwrite the old one (you're welcome to fix this with a PR).

6. Submit! You should get around 0.361 score on LB after 100 epochs.

## What's else inside?

* [Poor man's Exploration Data Analysis](https://github.com/killthekitten/kaggle-ds-bowl-2018-baseline/blob/master/visualize_inference.ipynb) -- to get a basic idea about data.
* [Test submit for errors](https://github.com/killthekitten/kaggle-ds-bowl-2018-baseline/blob/master/test_submission_for_errors.ipynb) -- tries to read the submission and visualizes separate masks.
* [Visualize inference](https://github.com/killthekitten/kaggle-ds-bowl-2018-baseline/blob/master/visualize_inference.ipynb) -- since there's not too many masks in the test dataset, it's easy to look through all of them in a single place.

## TODO

- [ ] Fix validation. For now, train data is used as a validation set.
- [ ] Normalize data.
- [ ] Move configuration to `argsparse` for easier hyperparameter search.
- [ ] Parallelize data loading.
- [ ] Augmentations.
- [ ] External Data.
