from torch.utils.data import Dataset, DataLoader
import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

batch_size = 16
input_size = (512, 512)



class HorseDataset(Dataset):
    def __init__(self, data_dir, gt_dir, input_size):
        self.data_dir = data_dir
        self.gt_dir = gt_dir
        self.input_size = input_size

        self.data_list = os.listdir(data_dir)
        self.gt_list = os.listdir(gt_dir)

        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.images = {}
        self.gts = {}
        self.labels = {}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        file_name = self.data_list[item]

        if file_name in self.images:
            image = self.images[file_name]
            gt = self.gts[file_name]
            label = self.labels[file_name]

        else:
            image_path = os.path.join(self.data_dir, file_name)
            gt_path = os.path.join(self.gt_dir, file_name)

            image = cv2.imread(image_path)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            image = cv2.resize(image, (self.input_size[0], self.input_size[1]), interpolation=cv2.INTER_CUBIC)
            # gt = cv2.resize(gt, (int(self.input_size[0] / 4), int(self.input_size[1] / 4)), interpolation=cv2.INTER_CUBIC)
            gt = cv2.resize(gt, (int(self.input_size[0]), int(self.input_size[1])), interpolation=cv2.INTER_CUBIC)
            gt = np.where(gt > 0, 1, 0)

            label = cv2.resize(label, (int(self.input_size[0]), int(self.input_size[1])), interpolation=cv2.INTER_CUBIC)
            label = np.where(label > 0, 1, 0)

            tmp = np.eye(2)[gt.reshape([-1])]
            seg_labels = tmp.reshape((int(self.input_size[0]), int(self.input_size[0]), 2))
            gt = seg_labels

            image = image.transpose(2, 1, 0)

            self.images.update({file_name: image})
            self.gts.update({file_name: gt})
            self.labels.update({file_name: label})

        image = torch.tensor(image, dtype=torch.float32)
        # image = self.transform(image)
        gt = torch.tensor(gt, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        sample = {
            'image': image,
            'gt': gt,
            'label': label
        }

        return sample


if __name__ == '__main__':
    data_dir = './data/rgb/'
    gt_dir = './data/figure_ground/'

    train_set = HorseDataset(data_dir, gt_dir, input_size)
    sampler_train = torch.utils.data.RandomSampler(train_set)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, batch_size, drop_last=True)
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train)

    for data in data_loader_train:
        gt = data['gt'][0].numpy()
        pass
