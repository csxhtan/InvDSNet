import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import csv
import glob
import random
import cv2


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, crop_size=(256, 256)):
        super(type(self), self).__init__()
        self.blurry_img_list = []
        self.clear_img_list = []
        self.snow_img_list = []
        self.crop_size = crop_size
        self.to_tensor = transforms.ToTensor()
        self.root = root

        self.nameclass = {}

        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.nameclass[name] = len(self.nameclass.keys())
        print(self.nameclass)
        self.blurry_img_list, self.clear_img_list, self.snow_img_list = self.load_csv('data.csv')

        if mode == 'train':
            self.blurry_img_list = self.blurry_img_list[:int(0.98 * len(self.blurry_img_list))]
            self.clear_img_list = self.clear_img_list[:int(0.98 * len(self.clear_img_list))]
            self.snow_img_list = self.snow_img_list[:int(0.98 * len(self.snow_img_list))]
        if mode == 'val':
            self.blurry_img_list = self.blurry_img_list[int(0.98 * len(self.blurry_img_list)):]
            self.clear_img_list = self.clear_img_list[int(0.98 * len(self.clear_img_list)):]
            self.snow_img_list = self.snow_img_list[int(0.98 * len(self.snow_img_list)):]
        print(len(self.blurry_img_list))

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.nameclass.keys():
                # 'all\\gt\\00001.jpg
                images += glob.glob(os.path.join(self.root, name, '*.tif'))

            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:  # 'all\\gt\\00000000.jpg'
                    name = img.split(os.sep)[-2]
                    category = self.nameclass[name]
                    # 'all\\gt\\00000000.png', 0
                    writer.writerow([img, category])
                print('writen into csv file:', filename)

        blurry_img, clear_img, snow_img = [], [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'all\\gt\\00000000.jpg', 0
                img, category = row
                category = int(category)
                if category == 0:
                    clear_img.append(img)
                elif category == 1:
                    snow_img.append(img)
                else:
                    blurry_img.append(img)

        assert len(clear_img) == len(snow_img) and len(snow_img) == len(blurry_img)

        return blurry_img, clear_img, snow_img

    def crop_resize_totensor(self, img, crop_location):
        img256 = img.crop(crop_location)
        img128 = img256.resize((self.crop_size[0] // 2, self.crop_size[1] // 2), resample=Image.BILINEAR)
        img64 = img128.resize((self.crop_size[0] // 4, self.crop_size[1] // 4), resample=Image.BILINEAR)
        return self.to_tensor(img256), self.to_tensor(img128), self.to_tensor(img64)

    def __len__(self):
        return len(self.clear_img_list)

    def __getitem__(self, idx):
        # filename processing
        blurry_img_name = self.blurry_img_list[idx]
        clear_img_name = self.clear_img_list[idx]
        snow_mask_name = self.snow_img_list[idx]

        blurry_left_img = cv2.imread(blurry_img_name, -1)
        clear_img = cv2.imread(clear_img_name, -1)
        snow_mask = cv2.imread(snow_mask_name, -1)

        blurry_left_img = self.to_tensor(blurry_left_img)
        clear_img = self.to_tensor(clear_img)
        snow_mask = self.to_tensor(snow_mask)

        blurry_left_img = torch.flip(blurry_left_img, dims=[0])
        clear_img = torch.flip(clear_img, dims=[0])
        snow_mask = torch.flip(snow_mask, dims=[0])

        blurry_left_img = transforms.ToPILImage()(blurry_left_img)
        clear_img = transforms.ToPILImage()(clear_img)
        snow_mask = transforms.ToPILImage()(snow_mask)

        assert blurry_left_img.size == clear_img.size and clear_img.size == snow_mask.size
        crop_left = int(np.floor(np.random.uniform(0, blurry_left_img.size[0] - self.crop_size[0] + 1)))
        crop_top = int(np.floor(np.random.uniform(0, blurry_left_img.size[1] - self.crop_size[1] + 1)))
        crop_location = (crop_left, crop_top, crop_left + self.crop_size[0], crop_top + self.crop_size[1])

        img256_left, img128_left, img64_left = self.crop_resize_totensor(blurry_left_img, crop_location)
        label256, label128, label64 = self.crop_resize_totensor(clear_img, crop_location)
        mask256, mask128, mask64 = self.crop_resize_totensor(snow_mask, crop_location)
        batch = {'img256': img256_left, 'img128': img128_left,
                 'img64': img64_left, 'label256': label256, 'label128': label128,
                 'label64': label64, 'mask256': mask256, 'mask128': mask128, 'mask64': mask64}
        for k in batch:
            batch[k] = batch[k] * 2 - 1.0  # in range [-1,1]
        return batch


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root, crop_size=(640, 480)):
        super(type(self), self).__init__()
        self.blurry_img_list = []
        self.clear_img_list = []
        self.snow_img_list = []
        self.crop_size = crop_size
        self.to_tensor = transforms.ToTensor()
        self.root = root

        self.nameclass = {}

        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.nameclass[name] = len(self.nameclass.keys())
        print(self.nameclass)
        self.blurry_img_list, self.clear_img_list, self.snow_img_list = self.load_csv('test_data.csv')

        print(len(self.blurry_img_list))

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.nameclass.keys():
                # 'all\\gt\\00001.jpg
                images += glob.glob(os.path.join(self.root, name, '*.tif'))

            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:  # 'all\\gt\\00000000.jpg'
                    name = img.split(os.sep)[-2]
                    category = self.nameclass[name]
                    # 'all\\gt\\00000000.png', 0
                    writer.writerow([img, category])
                print('writen into csv file:', filename)

        blurry_img, clear_img, snow_img = [], [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'all\\gt\\00000000.jpg', 0
                img, category = row
                category = int(category)
                if category == 0:
                    clear_img.append(img)
                elif category == 1:
                    snow_img.append(img)
                else:
                    blurry_img.append(img)

        assert len(clear_img) == len(snow_img) and len(snow_img) == len(blurry_img)

        return blurry_img, clear_img, snow_img

    def crop_resize_totensor(self, img, crop_location):
        img256 = img.crop(crop_location)
        img128 = img256.resize((self.crop_size[0] // 2, self.crop_size[1] // 2), resample=Image.BILINEAR)
        img64 = img128.resize((self.crop_size[0] // 4, self.crop_size[1] // 4), resample=Image.BILINEAR)
        return self.to_tensor(img256), self.to_tensor(img128), self.to_tensor(img64)

    def __len__(self):
        return len(self.clear_img_list)

    def __getitem__(self, idx):
        # filename processing
        blurry_img_name = self.blurry_img_list[idx]
        clear_img_name = self.clear_img_list[idx]
        snow_mask_name = self.snow_img_list[idx]

        blurry_left_img = cv2.imread(blurry_img_name, -1)
        clear_img = cv2.imread(clear_img_name, -1)
        snow_mask = cv2.imread(snow_mask_name, -1)

        blurry_left_img = self.to_tensor(blurry_left_img)
        clear_img = self.to_tensor(clear_img)
        snow_mask = self.to_tensor(snow_mask)

        blurry_left_img = torch.flip(blurry_left_img, dims=[0])
        clear_img = torch.flip(clear_img, dims=[0])
        snow_mask = torch.flip(snow_mask, dims=[0])

        blurry_left_img = transforms.ToPILImage()(blurry_left_img)
        clear_img = transforms.ToPILImage()(clear_img)
        snow_mask = transforms.ToPILImage()(snow_mask)

        assert blurry_left_img.size == clear_img.size and clear_img.size == snow_mask.size
        crop_left = int(np.floor(np.random.uniform(0, blurry_left_img.size[0] - self.crop_size[0] + 1)))
        crop_top = int(np.floor(np.random.uniform(0, blurry_left_img.size[1] - self.crop_size[1] + 1)))
        crop_location = (crop_left, crop_top, crop_left + self.crop_size[0], crop_top + self.crop_size[1])

        img256_left, img128_left, img64_left = self.crop_resize_totensor(blurry_left_img, crop_location)
        label256, label128, label64 = self.crop_resize_totensor(clear_img, crop_location)
        mask256, mask128, mask64 = self.crop_resize_totensor(snow_mask, crop_location)
        batch = {'img256': img256_left, 'img128': img128_left,
                 'img64': img64_left, 'label256': label256, 'label128': label128,
                 'label64': label64, 'mask256': mask256, 'mask128': mask128, 'mask64': mask64}
        for k in batch:
            batch[k] = batch[k] * 2 - 1.0  # in range [-1,1]
        return batch


if __name__ == '__main__':
    print('here')
