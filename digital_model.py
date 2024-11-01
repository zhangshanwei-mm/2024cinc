import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import albumentations as albu
import random
import os, sys
import numpy as np
from PIL import Image
import torch.nn.functional as F
import wfdb

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )   

class UNetModel(nn.Module):

    def __init__(self):
        super(UNetModel, self).__init__()
                
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, 14, 1) # 1x1 convolution

        
    def forward(self, x):
        conv1 = self.dconv_down1(x) # 64 x 512 x 512
        x = self.maxpool(conv1) # 64 x 256 x 256

        conv2 = self.dconv_down2(x) # 128 x 256 x 256
        x = self.maxpool(conv2) # 128 x 128 x 128
        
        conv3 = self.dconv_down3(x) # 256 x 128 x 128
        x = self.maxpool(conv3) # 256 x 64 x 64
        
        x = self.dconv_down4(x) # 512 x 64 x 64
        # print('before upsample', x.shape)
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)  
        
        x = self.dconv_up1(x) # 64 x 512 x 512
        
        out = self.conv_last(x) # 13 x 512 x 512

        return out


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        # inputs = inputs[:, 1:, :, :]
        # targets = targets[:, 1:, :, :]
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


class DiceMetric(nn.Module):
    def __init__(self, weight=None, size_average=True, include_background=False):
        super(DiceMetric, self).__init__()
        self.include_background = include_background

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        if self.include_background:
            inputs = inputs.view(-1)
            targets = targets.view(-1)
        else:
            inputs = inputs[:, 1:, :, :]
            targets = targets[:, 1:, :, :]
            inputs = inputs.contiguous().view(-1)
            targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return dice


def get_training_augmentation():
    train_transform = [
        albu.LongestMaxSize(max_size=512, always_apply=True),
        albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0, value=0),
        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=30, shift_limit=0.1, p=1, border_mode=0, value=0),
        albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0, value=0),
        
        albu.RandomCrop(height=500, width=500, always_apply=True),
        albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0, value=0),

        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.5,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.5,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.5,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.LongestMaxSize(max_size=512, always_apply=True),
        albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0, value=0),
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

class RescaleChannels(object):
    def __call__(self, sample):
        return 2 * sample - 1

class ECGImageDataset(Dataset):
    CLASSES = [
        'bg', 'I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'II(full)'
    ]
    def __init__(self, records, dir_root, augmentation=None, preprocessing=None):
        self.dir_root = dir_root
        self.records = records
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.data = self.load_data()

    def load_data(self):
        data = self.records
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        path_record = self.data[idx]
        path_record = os.path.join(self.dir_root, path_record)
        
        if os.path.exists(f"{path_record}-0wrinkles.png"):
            image_path = random.choice([f"{path_record}-0wrinkles.png", f"{path_record}-0.png"])
        else:
            image_path = f"{path_record}-0.png"
        mask_path = image_path.replace('.png', '.npz')
        info = {
            'path_record': path_record,
            'image_path': image_path,
            'mask_path': mask_path,
        }

        signals = self.__load_signal(path_record)
        # load image
        transform = transforms.Compose([
            transforms.Resize((850, 1100)), 
        ])
        image, bad_image_flag = self.__load_image(image_path, transform)
        image = np.array(image)
        # add shape info
        self.__add_shape_info(image, info)

        # load mask
        transform = transforms.Compose([
            transforms.Resize((850, 1100), Image.NEAREST), 
        ])
        mask, bad_mask_flag = self.__load_mask(mask_path, transform)
        mask = mask.permute(1, 2, 0).numpy()

        info['bad_flag'] = bad_image_flag or bad_mask_flag
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            image = torch.as_tensor(image, dtype=torch.float32)
            mask = torch.as_tensor(mask, dtype=torch.float32)
        else:
            image_transform = transforms.Compose([
                transforms.ToTensor(),
                RescaleChannels(),
            ])
            mask_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            image = image_transform(image)
            mask = mask_transform(mask)

        return image.float(), mask.float(), signals, info


    def __load_signal(self, path_record):
        try:
            signals, fields = wfdb.rdsamp(path_record)
        except Exception as e:
            signals = np.zeros((1000, 12))
        signals[np.isnan(signals)] = 0
        # sample signals to 1000
        if signals.shape[0] > 1000:
            signals_sampled = np.zeros((1000, signals.shape[1]))
            for i in range(signals.shape[1]):
                signal = signals[:, i]
                idx = np.round(np.linspace(0, len(signal) - 1, 1000)).astype(int)
                signal_sampled = signal[idx]
                signals_sampled[:, i] = signal_sampled
            signals = signals_sampled

        return np.float32(signals)

    def __load_image(self, image_path, transform=None):
        bad_image_flag = False
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(e)
            print(image_path)
            image = Image.new('RGB', (850, 1100))
            bad_image_flag = True

        if transform:
            image = transform(image)
        return image, bad_image_flag

    def __reformat_mask(self, mask):
        mask = torch.as_tensor(mask, dtype=torch.float32)
        mask = mask.permute(0, 2, 1)
        return mask

    def __multi_hot_encoding(self, mask, num_classes):
        mask = torch.as_tensor(mask, dtype=torch.long)
        background = torch.ones_like(mask[0])
        class_num = len(mask)
        for c in range(class_num):
            mask_tmp = mask[c]
            mask_tmp_up = torch.roll(mask[c], shifts=-1, dims=0)
            mask_tmp_down = torch.roll(mask[c], shifts=1, dims=0)
            mask_tmp_left = torch.roll(mask[c], shifts=-1, dims=1)
            mask_tmp_right = torch.roll(mask[c], shifts=1, dims=1)
            mask_tmp_up2 = torch.roll(mask[c], shifts=-2, dims=0)
            mask_tmp_down2 = torch.roll(mask[c], shifts=2, dims=0)
            mask[c] = mask_tmp | mask_tmp_up | mask_tmp_down | mask_tmp_left | mask_tmp_right | mask_tmp_up2 | mask_tmp_down2

        mask_flatten = torch.zeros_like(mask[0])
        for c in range(class_num):
            mask_flatten = mask_flatten | mask[c]
        background = background - mask_flatten
        background = background.unsqueeze(0)
        new_mask = torch.cat((background, mask), dim=0)

        return new_mask 
        
    def __load_mask(self, mask_path, transform=None):
        bad_mask_flag = False
        try:
            mask = np.load(mask_path)['mask'].astype(int)
        except Exception as e:
            print(e)
            print(mask_path)
            mask = np.zeros((13, 850, 1100))
            bad_mask_flag = True
        mask = self.__reformat_mask(mask)
        mask = self.__multi_hot_encoding(mask, 14)
        mask = transform(mask)
        return mask, bad_mask_flag

    def __add_shape_info(self, image, info):
        im_h, im_w, c= image.shape # np.array image.shape[1], image.shape[0]
        larger_dim = max(im_w, im_h)
        # resize image width to 2048 with equal proportion,then pad image to 2048x2048
        pad_width = larger_dim - im_w
        pad_height = larger_dim - im_h

        info['larger_dim'] = larger_dim
        info['pad_width'] = pad_width
        info['pad_height'] = pad_height
