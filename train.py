from FPN import FPN
from unet import Unet
import torch
import datetime
import random
from data import *
import time
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from loss import seg_loss
from FPUnet import FPUnet

seed = 0
device = torch.device('cuda')
lr = 0.001
lr_drop = 25  # 经过一段的时间后学习率下降
optimizer = 'Adam'
batch_size = 4

data_dir = './data_all/rgb/'
gt_dir = './data_all/figure_ground/'
tensorboard_dir = './runs/'
log_dir = './logs/'
input_size = (512, 512)

start_epoch = 0
epochs = 150

resume = False
ckpt = './ckpt/epoch_50_0818.pth'
ckpt_dir = './ckpt/'

if __name__ == '__main__':
    model = Unet()
    torch.manual_seed(seed)
    random.seed(seed)
    model.to(device)

    fpn = FPN([3,4,23,3])
    fpn.load_state_dict(state_dict=torch.load('./ckpt/epoch_40_fpn.pth', map_location=torch.device('cpu'))['model'])
    fpn = fpn.to(device).eval()

    # 参数分析
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    t = model.named_parameters()
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": lr,
        },
    ]

    # 优化器设置
    if optimizer == 'SGD' or optimizer == None:
        optimizer = torch.optim.SGD(param_dicts, lr=lr)
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(param_dicts, lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop)

    # 数据集设置
    train_set = HorseDataset(data_dir, gt_dir, input_size)
    sampler_train = torch.utils.data.RandomSampler(train_set)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, batch_size, drop_last=True)
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train)

    if resume:
        checkpoint = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    print("Start training")
    start_time = time.time()
    run_log_name = 'exper_{:.4f}'.format(start_time)
    run_log_name = os.path.join(log_dir, run_log_name)
    # save the performance during the training
    writer = SummaryWriter(tensorboard_dir)
    criterion = torch.nn.SmoothL1Loss()
    criterion = torch.nn.MSELoss()
    criterion = seg_loss

    loss_test_list = []
    loss_epoch = []
    step = 0
    writer = SummaryWriter(tensorboard_dir)
    for epoch in range(start_epoch, epochs):
        t1 = time.time()
        running_loss = 0
        n = 0
        model.train()
        for index, data in enumerate(data_loader_train):
            n = n + 1
            input = data['image']
            gt = data['gt']
            label = data['label']
            input = input.to(device)
            gt = gt.to(device)
            label = label.to(device)

            enhance, p3, p4, p5 = fpn(input)
            enhance = F.upsample(enhance, size=input_size, mode='bilinear')
            enhance = enhance * 100
            input = input + enhance

            p2 = model(input)

            # p2 = F.upsample(p2,size=input_size,mode='bilinear')

            loss = criterion(p2, gt ,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss_epoch.append(running_loss / n)
        t2 = time.time()
        if writer is not None:
            with open(run_log_name, 'a') as log_file:
                log_file.write("{} epoch loss:{:.5f}, cost time: {:.2f}s \n".format(epoch, loss_epoch[epoch], t2 - t1))
            writer.add_scalar('loss/loss', loss_epoch[epoch], epoch)
        print("{} epoch loss:{:.5f}, cost time: {} s ".format(epoch, loss_epoch[epoch], t2 - t1))
        lr_scheduler.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            ckpt_name = 'epoch_{:d}_0818.pth'.format(epoch+51)
            checkpoint_latest_path = os.path.join(ckpt_dir, ckpt_name)
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch
            }, checkpoint_latest_path)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
