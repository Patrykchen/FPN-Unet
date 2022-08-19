import torch
from FPN import FPN
from data import HorseDataset
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from unet import Unet
from tqdm import tqdm
import torch.nn.functional as F
from analyze import get_miou,boundary_iou
import random

data_dir = './datatest/rgb/'
gt_dir = './datatest/figure_ground/'
tensorboard_dir = './runs/'
log_dir = './logs/'
result_dir = './result/'
input_size = (512, 512)
seed = 0

batch_size = 1

device = 'cuda'

if __name__ == '__main__':
    torch.manual_seed(seed)
    random.seed(seed)

    model = Unet()
    model.load_state_dict(state_dict=torch.load('./ckpt/epoch_99_enhancemerge2.pth', map_location=torch.device('cpu'))['model'])
    model = model.to(device).eval()

    fpn = FPN([3, 4, 23, 3])
    fpn.load_state_dict(state_dict=torch.load('./ckpt/epoch_40_fpn.pth', map_location=torch.device('cpu'))['model'])
    fpn = fpn.to(device).eval()

    miou = 0
    biou = 0

    with torch.no_grad():
        test_set = HorseDataset(data_dir, gt_dir, input_size)
        sampler_train = torch.utils.data.SequentialSampler(test_set)
        batch_sampler_test = torch.utils.data.BatchSampler(
            sampler_train, batch_size, drop_last=True)
        data_loader_test = DataLoader(test_set, batch_sampler=batch_sampler_test)

        for batch_idx, data in enumerate(tqdm(data_loader_test)):
            input = data['image']
            gt = data['label']
            input = input.to(device)

            enhance, p3, p4, p5 = fpn(input)
            enhance = F.upsample(enhance, size=input_size, mode='bilinear')
            enhance = enhance * 100
            input = input + enhance

            gt = gt.to(device)
            gt = gt.cpu().numpy()
            p2 = model(input)
            p2 = p2.cpu().numpy()
            p2 = np.where(p2[:,0]>p2[:,1],0,1)
            p2 = p2.reshape(input_size)
            miou += get_miou(gt,p2)
            biou += boundary_iou(gt,p2)

            # p2 = F.upsample(p2, size=input_size, mode='bilinear')

            # p2 = (p2.squeeze(0)).squeeze(0)

            # result = np.where(result > 0.5, 255, 0)
            p2 *= 255
            image_name = str(batch_idx)
            image_name = image_name + '.jpg'
            cv2.imwrite(os.path.join(result_dir, image_name), p2)

    print(miou/batch_idx)
    print(biou/batch_idx)
