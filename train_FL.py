import pandas as pd
import argparse
import json
from PIL import Image

import torchvision.transforms as transforms
import torch

#local imports
from chexpert_data import CheXpertDataSet


IMAGENET_MEAN = [0.485, 0.456, 0.406]  # mean of ImageNet dataset(for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]   # std of ImageNet dataset(for normalization)

#TODO configure resource usage
use_gpu = torch.cuda.is_available()

def main():

    #parse config file
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('cfg_path', metavar='CFG_PATH', type = str, help = 'Path to the config file in json format.')
    args = parser.parse_args()
    with open(args.cfg_path) as f:
        cfg = json.load(f)

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






if __name__ == "__main__":
    main()
