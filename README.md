Image Desnowing via Deep Invertible Separation
=
This is the official PyTorch implementation of the TCSVT 2023 Paper.
[Here is the link of the paper.](https://ieeexplore.ieee.org/abstract/document/10004991)

Data
=
Put the data you need in the data folder.

Train
=
Set the training configs in train_config.py;
Set the root of the training data and cropping size in line 63 and 64 of train.py;
If you want to continue training on a trained model, remember to reset the resume_epoch in train_config.py.

Test
=
