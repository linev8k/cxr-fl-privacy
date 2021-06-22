"""
This script is used for getting the mean and std for our own dataset.
"""
#https://github.com/wll199566/CheXpert/blob/master/statistics.py

import numpy as np
import pandas as pd
from skimage import io

import torch
import torchvision
from torchvision import transforms

# get the path for train_csv file and valid_csv file
csv_root_path = "/hpi/fs00/share/fg-arnrich/datasets/CheXpert-v1.0-small/"
train_csv_filename = "train.csv"
valid_csv_filename = "valid.csv"

# read the dataset
with open(csv_root_path+train_csv_filename, "rt") as fin:
    train_df = pd.read_csv(fin)

with open(csv_root_path+valid_csv_filename, "rt") as fin:
    valid_df = pd.read_csv(fin)

# get the path
image_root_path = "/hpi/fs00/share/fg-arnrich/datasets/"
train_image_filenames = np.asarray(train_df.iloc[:, 0])
valid_image_filenames = np.asarray(valid_df.iloc[:, 0])

# use the transform to resize the images
train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([364, 364]),
        transforms.ToTensor()])


# get the statistics for the training dataset
mean_train = 0.0
std_train = 0.0
num_samples_train = 0.0
for train_image_file in train_image_filenames:
    image = io.imread(image_root_path+train_image_file)
    image = np.repeat(image[None,...], 3, axis=0).transpose(1, 2, 0)
    image = train_transform(image)
    image = image.view(image.size(0), -1)
    mean_train += image.float().mean(1)
    std_train += image.float().std(1)
    num_samples_train += 1

print("mean for training dataset", mean_train / num_samples_train)
print("std for training dataset", std_train / num_samples_train)

# get the statistics for the validation dataset
mean_valid = 0.0
std_valid = 0.0
num_samples_valid = 0.0
for valid_image_file in valid_image_filenames:
    image = io.imread(image_root_path+valid_image_file)
    image = np.repeat(image[None,...], 3, axis=0).transpose(1, 2, 0)
    image = train_transform(image)
    image = image.view(image.size(0), -1)
    mean_valid += image.float().mean(1)
    std_valid += image.float().std(1)
    num_samples_valid += 1

print("mean for validation dataset", mean_valid / num_samples_valid)
print("std for validation dataset", std_valid / num_samples_valid)
