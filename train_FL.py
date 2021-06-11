
"""Train a model using federated learning"""

#set which GPUs to use
import os
selected_gpus = [0,1] #configure this
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in selected_gpus])

import pandas as pd
import argparse
import json
from PIL import Image
import time

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
use_gpu = torch.cuda.is_available()

#local imports
from chexpert_data import CheXpertDataSet
from trainer import Trainer, DenseNet121
from utils import check_path


IMAGENET_MEAN = [0.485, 0.456, 0.406]  # mean of ImageNet dataset(for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]   # std of ImageNet dataset(for normalization)


def main():

    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #parse config file
    parser.add_argument('cfg_path', type = str, help = 'Path to the config file in json format.')
    #output path for storing results
    parser.add_argument('--output_path', '-o', help = 'Path to save results.', default = 'results/')
    #whether to assert GPU usage (disable for testing without GPU)
    parser.add_argument('--no_gpu', dest='no_gpu', help='Don\'t verify GPU usage.', action='store_true')
    #set path to chexpert data
    parser.add_argument('--chexpert', '-d', dest='chexpert_path', help='Path to CheXpert data.', default='./')
    args = parser.parse_args()
    with open(args.cfg_path) as f:
        cfg = json.load(f)

    if not args.no_gpu:
        check_gpu_usage(use_gpu)

    #only use pytorch randomness, check for pitfalls when using other modules
    random_seed = cfg['random_seed']
    if random_seed != None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    # Parameters from config file
    nnIsTrained = cfg['pre_trained']     # pre-trained using ImageNet
    trBatchSize = cfg['batch_size']
    trMaxEpoch = cfg['max_epochs']

    # Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = cfg['imgtransResize']
    # imgtransCrop = cfg['imgtransCrop']
    policy = cfg['policy']

    nnClassCount = cfg['nnClassCount']       # dimension of the output
    class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

    #run preprocessing to obtain these files
    data_path = check_path(args.chexpert_path, warn_exists=False, require_exists=True)
    pathFileTrain = data_path + 'CheXpert-v1.0-small/train_mod.csv'
    pathFileValid =  data_path + 'CheXpert-v1.0-small/valid_mod.csv'
    pathFileTest = data_path + 'CheXpert-v1.0-small/test_mod.csv'


    # define transforms
    # if using augmentation, use different transforms for training, test & val data
    transformSequence = transforms.Compose([transforms.Resize((imgtransResize,imgtransResize)),
                                            # transforms.RandomResizedCrop(imgtransResize),
                                            # transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            # transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
                                            ])

    # Load dataset
    datasetTrain = CheXpertDataSet(data_path, pathFileTrain, nnClassCount, policy, transform = transformSequence)
    len_train = len(datasetTrain)
    print("Train data length:", len_train)

    #remove transformations here?
    datasetValid = CheXpertDataSet(data_path, pathFileValid, nnClassCount, policy, transform = transformSequence)
    print("Valid data length:", len(datasetValid))

    datasetTest = CheXpertDataSet(data_path, pathFileTest, nnClassCount, policy, transform = transformSequence)
    print("Test data length:", len(datasetTest))

    # assert datasetTrain[0][0].shape == torch.Size([3,imgtransResize,imgtransResize])
    # assert datasetTrain[0][1].shape == torch.Size([nnClassCount])


    #Create dataLoaders, normal training
    dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True, num_workers=2, pin_memory=True)
    dataLoaderVal = DataLoader(dataset = datasetValid, batch_size = trBatchSize, num_workers = 2, pin_memory = True)
    dataLoaderTest = DataLoader(dataset = datasetTest, num_workers = 2, pin_memory = True)
    print('Length train dataloader (n batches): ', len(dataLoaderTrain))

    #show images for testing
    # for batch in dataLoaderTrain:
    #     transforms.ToPILImage()(batch[0][0]).show()


    if use_gpu:
        model = DenseNet121(nnClassCount, cfg['pre_trained']).cuda()
    else:
        model = DenseNet121(nnClassCount, cfg['pre_trained'])

    output_path = check_path(args.output_path, warn_exists=True)



    # start = time.time()
    # model_num, params = Trainer.train(model, dataLoaderTrain, dataLoaderVal, nnClassCount, cfg, output_path, use_gpu)
    # end = time.time()
    # print(f"Total time: {end-start}")

    # outGT, outPRED = Trainer.test(model, dataLoaderTest, nnClassCount, class_names, use_gpu,
    #                                     checkpoint= output_path+'1-epoch_FL.pth.tar')



def check_gpu_usage(use_gpu):
    assert use_gpu is True, "GPU not used"
    assert torch.cuda.device_count() == len(selected_gpus), "Wrong number of GPUs available to Pytorch"
    print(f"{torch.cuda.device_count()} GPUs available")

    return True




if __name__ == "__main__":
    main()
