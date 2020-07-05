from src.data.datasets import BaseDataset
import os
import pandas as pd
import numpy as np
import torch
# from PIL import Image
import cv2
from src.data.transforms import Normalize
# from src.data.transforms_v2 import get_transforms

# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

class No_Aug_Dataset(BaseDataset):
    def __init__(self, data_dir, mean=None, std=None, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.transforms = Normalize(mean, std)

        if self.type == 'train' or self.type == 'valid': 
            if self.type == 'train': file_path = os.path.join(data_dir, 'split_train.csv')
            else: file_path = os.path.join(data_dir, 'split_valid.csv')
            
            df = pd.read_csv(file_path, names=['filename', 'category'])
            self.category = torch.tensor(df['category'].values).to(torch.int64)
            self.filename = df['filename'].values
        else:
            file_path = os.path.join(data_dir, 'test.csv')
            df = pd.read_csv(file_path, names=['filename', 'category'], header=None)
            self.filename = df['filename'].values[1:]

    def __getitem__(self, index):
        if self.type == 'train' or self.type == 'valid':
            label = self.category[index]
            img_dir_path = os.path.join(self.data_dir, 'train/train/')
            img_name = self.filename[index]
            img_path = os.path.join(img_dir_path, str(label.item()).zfill(2), img_name)

            # img = Image.open(img_path).convert('RGB')
            img = cv2.imread(img_path)
            img = cv2.resize(img, (600, 600), interpolation=cv2.INTER_AREA)
            # img = self.transforms(np.array(img))
            img = torch.from_numpy(img).to(torch.float)
            img = img.permute(2,0,1)
            return {'img': img, 'label': label}
        else: 
            img_dir_path = os.path.join(self.data_dir, 'test/test/')
            img_name = self.filename[index]
            img_path = os.path.join(img_dir_path, img_name)
            # print(img_path)
            # img = Image.open(img_path).convert('RGB')
            # img = self.transforms['val_test'](img)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (600,600), interpolation=cv2.INTER_AREA)
            # img = self.transforms(img)
            img = torch.from_numpy(img).to(torch.float)
            img = img.permute(2,0,1)
            return {'img': img, 'filename': self.filename[index]}

    def __len__(self):
        return len(self.filename)

# path = '/nfs/nas-5.1/wbcheng/shopee_task2'

# dataset = ImageDataset(path, type_='test')
# train_sampler = RandomSampler(dataset)
# train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=8)

# for batch in train_dataloader:
#     print(batch['img'])
    