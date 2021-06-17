"""Utility methods"""

import os
from chexpert_data import CheXpertDataSet

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

def check_path(path, warn_exists=True, require_exists=False):

    """Check path to directory.
        warn_exists: Warns and requires validation by user to use the specified path if it already exists.
        require_exists: Aborts if the path does not exist. """

    if path[-1] != '/':
        path = path + '/'

    create_path = True

    if os.path.exists(path):
        create_path = False
        if warn_exists:
            replace = ''
            while replace not in ['y', 'n']:
                replace = input(f"Path {path} already exists. Files may be replaced. Continue? (y/n): ")
                if replace == 'y':
                    pass
                elif replace == 'n':
                    exit('Aborting, run again with a different path.')
                else:
                    print("Invalid input")


    if require_exists:
        if not os.path.exists(path):
            exit(f"{path} does not exist. Aborting")

    if create_path:
        os.mkdir(path)
        print(f"Created {path}")

    return path

def get_mean_std(data_path='./', csv_file='train.csv'):

    #https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/2
    """Calculates mean and standard deviation of CheXpert data.
    Args:
        data_path: Where data lives. Directory that contains 'CheXpert-v1.0-small/'.
        csv_file: CSV containing subset of images that should be taken into account for calculation.
    Note: This returns 3x the same value for mean and std, because the images are greyscale.
    It's necessary to use this however when making use of a pretrained network with a defined input shape.
    """

    #dummy variables
    class_idx=[0]
    policy='zeros'

    pathFileTrain = data_path + 'CheXpert-v1.0-small/' + csv_file
    dataset = CheXpertDataSet(data_path, pathFileTrain, class_idx, policy, transform = transforms.ToTensor())

    dataloader = DataLoader(dataset=dataset, batch_size=50, shuffle=False)

    mean = 0.0
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        print(images)
    mean = mean / len(dataset)

    var = 0.0
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(var / (len(dataset)*images.size(2)))

    return mean, std
