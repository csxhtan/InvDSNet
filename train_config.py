# train_config
test_time = False

train = {}
train['train_img_list'] = './train_gopro_gamma.txt'
train['val_img_list'] = './val_gopro_gamma.txt'
train['batch_size'] = 8
train['val_batch_size'] = 8
train['num_epochs'] = 2000
train['log_epoch'] = 1
train['optimizer'] = 'Adam'
train['learning_rate'] = 1e-4
# -- for SGD -- #
train['momentum'] = 0.9
train['nesterov'] = True

# config for save , log and resume
train['sub_dir'] = 'replicate'
train['resume'] = './save/replicate/0'
train['resume_epoch'] = None  # None means the last epoch
train['resume_optimizer'] = './save/replicate/0'

net = {}
net['xavier_init_all'] = True

loss = {}
loss['weight_l2_reg'] = 0.0
