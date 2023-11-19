import torch
import torch.nn as nn
import torchvision
from InvDSNet import InvDDNet
import numpy as np
from log import TensorBoardX
from utils import *
import train_config as config
from data import Dataset
from functools import partial
from tqdm import tqdm
from time import time
import copy
import sys

log10 = np.log(10)
MAX_DIFF = 2


# constant_matrix = (torch.zeros((8, 3, 128, 128))).cuda()


def compute_loss(db256, fuse_db, f, h, g, batch, rev=False):
    assert db256.shape[0] == batch['label256'].shape[0]

    loss = 0
    if rev:
        loss += mse(db256[:, :3], batch['img256'])
        psnr = 10 * torch.log(MAX_DIFF ** 2 / loss) / log10
        loss = loss * 0.5
        loss += mse(fuse_db[:, :3], batch['img256']) * 0.5
        m = batch['mask64'][:, :1, :, :]
        loss += 0.1 * mse(f, m)
        loss += 0.1 * mse(h, m)
        loss += 0.1 * mse(g, m)
    else:
        loss += mse(db256[:, :3], batch['label256'])
        psnr = 10 * torch.log(MAX_DIFF ** 2 / loss) / log10
        loss = loss * 10
        loss += mse(fuse_db[:, :3], batch['mask256'])
        m = batch['mask64'][:, :1, :, :]
        loss += 0.5 * mse(f, m)
        loss += 0.5 * mse(h, m)
        loss += 0.5 * mse(g, m)

    return {'mse': loss, 'psnr': psnr}


def backward(loss, optimizer):
    loss['mse'].backward(retain_graph=True)

    return


def set_learning_rate(optimizer, epoch):
    optimizer.param_groups[0]['lr'] = config.train['learning_rate']  # * 0.3 ** (epoch // 500)


if __name__ == "__main__":
    tb = TensorBoardX(config_filename='train_config.py', sub_dir=config.train['sub_dir'])
    log_file = open('{}/{}'.format(tb.path, 'train.log'), 'w')

    train_dataset = Dataset('./data/CSD/Train', mode='train', crop_size=(256, 256))
    val_dataset = Dataset('./data/CSD/Train', mode='val', crop_size=(256, 256))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train['batch_size'], shuffle=True,
                                                   drop_last=True, num_workers=8, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.train['val_batch_size'], shuffle=True,
                                                 drop_last=True, num_workers=8, pin_memory=True)

    mse = torch.nn.MSELoss().cuda()
    l1 = torch.nn.L1Loss().cuda()
    net = torch.nn.DataParallel(InvDDNet()).cuda()

    assert config.train['optimizer'] in ['Adam', 'SGD']
    if config.train['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=config.train['learning_rate'],
                                     weight_decay=config.loss['weight_l2_reg'])
    if config.train['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=config.train['learning_rate'],
                                    weight_decay=config.loss['weight_l2_reg'], momentum=config.train['momentum'],
                                    nesterov=config.train['nesterov'])

    last_epoch = -1

    if config.train['resume'] is not None:
        last_epoch = load_model(net, config.train['resume'], epoch=config.train['resume_epoch'])

    if config.train['resume_optimizer'] is not None:
        _ = load_optimizer(optimizer, net, config.train['resume_optimizer'], epoch=config.train['resume_epoch'])
        assert last_epoch == _

    train_loss_log_list = []
    val_loss_log_list = []
    first_val = True

    t = time()
    best_val_psnr = 0
    best_net = None
    best_optimizer = None

    for epoch in tqdm(range(last_epoch + 1, config.train['num_epochs']), file=sys.stdout):
        set_learning_rate(optimizer, epoch)
        tb.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch * len(train_dataloader), 'train')
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), file=sys.stdout,
                                desc='training'):
            t_list = []
            for k in batch:
                batch[k] = batch[k].cuda(non_blocking=True)
                batch[k].requires_grad = False

            optimizer.zero_grad()
            x_input = batch['img256']
            t = time()
            db256, fuse_db, f, h, g, tt = net(x_input, x_input)
            loss = compute_loss(db256, fuse_db, f, h, g, batch)
            backward(loss, optimizer)
            temp = loss
            # rb256, fuse_rb, f, h, g, _ = net(batch['label256'], batch['mask256'], rev=True)
            # loss = compute_loss(rb256, fuse_rb, f, h, g, batch, rev=True)
            # backward(loss, optimizer)
            optimizer.step()
            print(tt - t, end='')
            loss = temp

            for k in loss:
                loss[k] = float(loss[k].cpu().detach().numpy())
            train_loss_log_list.append({k: loss[k] for k in loss})
            for k, v in loss.items():
                tb.add_scalar(k, v, epoch * len(train_dataloader) + step, 'train')

        # validate and log
        if first_val or epoch % config.train['log_epoch'] == config.train['log_epoch'] - 1:
            with torch.no_grad():
                first_val = False
                for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), file=sys.stdout,
                                        desc='validating'):
                    for k in batch:
                        batch[k] = batch[k].cuda(non_blocking=True)
                        batch[k].requires_grad = False
                    x_input = batch['img256']
                    db256, fuse_db, f, h, g, _ = net(x_input, x_input)
                    loss = compute_loss(torch.clamp(db256, min=-1.0, max=1.0), fuse_db, f, h, g, batch)
                    for k in loss:
                        loss[k] = float(loss[k].cpu().detach().numpy())
                    val_loss_log_list.append({k: loss[k] for k in loss})

                train_loss_log_dict = {k: float(np.mean([dic[k] for dic in train_loss_log_list])) for k in
                                       train_loss_log_list[0]}
                val_loss_log_dict = {k: float(np.mean([dic[k] for dic in val_loss_log_list])) for k in
                                     val_loss_log_list[0]}
                for k, v in val_loss_log_dict.items():
                    tb.add_scalar(k, v, (epoch + 1) * len(train_dataloader), 'val')
                if best_val_psnr < val_loss_log_dict['psnr'] + 0.5:
                    best_val_psnr = val_loss_log_dict['psnr']
                    save_model(net, tb.path, epoch)
                    save_optimizer(optimizer, net, tb.path, epoch)
                if epoch % 50 == 0:
                    save_model(net, tb.path, epoch)
                    save_optimizer(optimizer, net, tb.path, epoch)

                train_loss_log_list.clear()
                val_loss_log_list.clear()

                tt = time()
                log_msg = ""
                log_msg += "epoch {} , {:.2f} imgs/s".format(epoch, (
                        config.train['log_epoch'] * len(train_dataloader) * config.train['batch_size'] + len(
                    val_dataloader) * config.train['val_batch_size']) / (tt - t))

                log_msg += " | train : "
                for idx, k_v in enumerate(train_loss_log_dict.items()):
                    k, v = k_v
                    if k == 'acc':
                        log_msg += "{} {:.3%} {}".format(k, v, ',')
                    else:
                        log_msg += "{} {:.5f} {}".format(k, v, ',')
                log_msg += "  | val : "
                for idx, k_v in enumerate(val_loss_log_dict.items()):
                    k, v = k_v
                    if k == 'acc':
                        log_msg += "{} {:.3%} {}".format(k, v, ',')
                    else:
                        log_msg += "{} {:.5f} {}".format(k, v, ',' if idx < len(val_loss_log_list) - 1 else '')
                tqdm.write(log_msg, file=sys.stdout)
                sys.stdout.flush()
                log_file.write(log_msg + '\n')
                log_file.flush()
                t = time()
                # print( torch.max( predicts , 1  )[1][:5] )

            # train_loss_epoch_list = []
