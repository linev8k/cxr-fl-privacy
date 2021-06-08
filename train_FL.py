
"""Train a model using federated learning"""

#set which GPUs to use
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3' #configure this

import pandas as pd
import argparse
import json
from PIL import Image

import torch
import torchvision.transforms as transforms
use_gpu = torch.cuda.is_available()

#local imports
from chexpert_data import CheXpertDataSet


IMAGENET_MEAN = [0.485, 0.456, 0.406]  # mean of ImageNet dataset(for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]   # std of ImageNet dataset(for normalization)


def main():

    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #parse config file
    parser.add_argument('cfg_path', type = str, help = 'Path to the config file in json format.')
    #whether to assert GPU usage (disable for testing without GPU)
    parser.add_argument('--no_gpu', dest='no_gpu', help='Don\'t verify GPU usage.', action='store_true')
    args = parser.parse_args()
    with open(args.cfg_path) as f:
        cfg = json.load(f)

    if not args.no_gpu:
        check_gpu_usage(use_gpu)

    #TODO configure randomness
    random_seed = cfg['random_seed']
    torch.manual_seed(random_seed)

    # Parameters from config file
    nnIsTrained = cfg['pre_trained']     # pre-trained using ImageNet
    trBatchSize = cfg['batch_size']
    trMaxEpoch = cfg['epochs']

    # Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = cfg['imgtransResize']
    # imgtransCrop = cfg['imgtransCrop']
    policy = cfg['policy']

    nnClassCount = cfg['nnClassCount']       # dimension of the output
    class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

    #run preprocessing to obtain these files
    pathFileTrain = './CheXpert-v1.0-small/train_mod.csv'
    pathFileValid = './CheXpert-v1.0-small/valid_mod.csv'
    pathFileTest = './CheXpert-v1.0-small/test_mod.csv'

    # define transforms
    transformSequence = transforms.Compose([transforms.Resize((imgtransResize,imgtransResize)),
                                            # transforms.RandomResizedCrop(imgtransCrop),
                                            # transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
                                            ])

    # Load dataset
    datasetTrain = CheXpertDataSet(pathFileTrain, nnClassCount, policy, transform = transformSequence)
    print("Train data length:", len(datasetTrain))

    datasetValid = CheXpertDataSet(pathFileValid, nnClassCount, policy, transform = transformSequence)
    print("Valid data length:", len(datasetValid))

    datasetTest = CheXpertDataSet(pathFileTest, nnClassCount, policy, transform = transformSequence)
    print("Test data length:", len(datasetTest))

    assert datasetTrain[0][0].shape == torch.Size([3,imgtransResize,imgtransResize])
    assert datasetTrain[0][1].shape == torch.Size([nnClassCount])



def check_gpu_usage(use_gpu):
    assert use_gpu is True, "GPU not used"
    assert torch.cuda.device_count() == len(os.environ["CUDA_VISIBLE_DEVICES"]), "Wrong number of GPUs available to Pytorch"
    print(f"{torch.cuda.device_count} GPUs available")

    return True


if __name__ == "__main__":
    main()
