"""
Get the mean and std for CheXpert data.
"""
#https://github.com/wll199566/CheXpert/blob/master/statistics.py

#output
# mean for training dataset tensor([0.5029, 0.5029, 0.5029])
# std for training dataset tensor([0.2897, 0.2897, 0.2897])
# mean for validation dataset tensor([0.5028, 0.5028, 0.5028])
# std for validation dataset tensor([0.2899, 0.2899, 0.2899])

import numpy as np
import pandas as pd
from skimage import io

import torch
import torchvision
from torchvision import transforms

# get the path for train_csv file and valid_csv file
csv_root_path = "/hpi/fs00/share/fg-arnrich/datasets/xray_FL/CheXpert-v1.0-small/"
train_csv_filename = "train.csv"
valid_csv_filename = "valid.csv"

# read the dataset
with open(csv_root_path+train_csv_filename, "rt") as fin:
    train_df = pd.read_csv(fin)

with open(csv_root_path+valid_csv_filename, "rt") as fin:
    valid_df = pd.read_csv(fin)

# get the path
image_root_path = "/hpi/fs00/share/fg-arnrich/datasets/xray_FL/"
train_image_filenames = np.asarray(train_df.iloc[:, 0])
valid_image_filenames = np.asarray(valid_df.iloc[:, 0])

# use the transform to resize the images
train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([364, 364]),
        transforms.ToTensor()])

#using a random 10% subset of training data
print(len(train_image_filenames))
train_image_filenames = random.sample(list(train_image_filenames), int(0.1*len(train_image_filenames)))
print(len(train_image_filenames))
print(train_image_filenames[:10])

# get the statistics for the training dataset
mean_train = 0.0
std_train = 0.0
num_samples_train = 0.0
img_count = 0
for train_image_file in train_image_filenames:
    image = io.imread(image_root_path+train_image_file)
    image = np.repeat(image[None,...], 3, axis=0).transpose(1, 2, 0)
    image = train_transform(image)
    image = image.view(image.size(0), -1)
    mean_train += image.float().mean(1)
    std_train += image.float().std(1)
    num_samples_train += 1

    if (num_samples%100==0):
        img_count += 100
        print(f"{img_count} images processed")

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
