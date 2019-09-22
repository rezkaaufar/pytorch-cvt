# pytorch-cvt
Cross view training for sequence labeling in pytorch

This repository contains my re-implementation of Cross View Training (https://arxiv.org/abs/1809.08370). At the moment, the implementation is only available for sequence labeling tasks.

## Load your dataset
Put your dataset in a folder and specify it with the command `--train_path`, `--dev_path`, and `--test_path`. Note that the implementation requires the dataset to be in a two column CoNLL format.

For the unlabeled dataset, specify it with the command `--unlabaled_path`. The tokens should be separated with a space and sentences are separated with a space.

## To do
- Tests