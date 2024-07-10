Image Desnowing via Deep Invertible Separation
=
This is the official PyTorch implementation of the TCSVT 2023 paper.

Data
-
Put the training data you need under the directory 'data'.

Rename the directories that contains snowy images, clean images and snow masks to 'Snow', 'Gt' and 'Mask'.

Train
-
Set the training configs in train_config.py;

Set the root of the training data and cropping size in train.py by function Dataset();

If you want to continue training on a trained model, remember to reset the resume, resume_epoch and resume_optimizer in train_config.py.

Test
-
Set your traied model in test.py by function load_model();

Set the root of your testing data in test.py.

Pretrained model
-
We provide the pretrained model trained on CSD, and you could use it for testing. See the link below (pwd: invd):

[Download Pretrained Model](https://pan.baidu.com/s/1AWxyk7XUpu83kMIKYz15uA)

Citation
-
    @article{quan2023image,
      title={Image desnowing via deep invertible separation},
      author={Quan, Yuhui and Tan, Xiaoheng and Huang, Yan and Xu, Yong and Ji, Hui},
      journal={IEEE Transactions on Circuits and Systems for Video Technology},
      year={2023},
      publisher={IEEE}
    }

Contacts
-
If you have questions, please contact with csxiaohengtan@foxmail.com
