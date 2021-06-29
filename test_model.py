
"""Train a model using federated learning"""

#set which GPUs to use
import os
selected_gpus = [6,7] #configure this
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in selected_gpus])

import argparse
import json
from PIL import Image
import time
import random
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#local imports
from chexpert_data import CheXpertDataSet
from trainer import Trainer, DenseNet121, Client
from utils import check_path


IMAGENET_MEAN = [0.485, 0.456, 0.406]  # mean of ImageNet dataset(for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]   # std of ImageNet dataset(for normalization)

CHEXPERT_MEAN = [0.5029, 0.5029, 0.5029]
CHEXPERT_STD = [0.2899, 0.2899, 0.2899]


def main():
    use_gpu = torch.cuda.is_available()

    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #parse config file
    parser.add_argument('cfg_path', type = str, help = 'Path to the config file in json format.')
    parser.add_argument('--no_gpu', dest='no_gpu', help='Don\'t verify GPU usage.', action='store_true')
    #model checkpoint path
    parser.add_argument('--model', '-m', dest='model_path', help='Path to model.')
    #set path to chexpert data
    parser.add_argument('--chexpert', '-d', dest='chexpert_path', help='Path to CheXpert data.', default='./')

    args = parser.parse_args()
    with open(args.cfg_path) as f:
        cfg = json.load(f)

    if not args.no_gpu:
        check_gpu_usage(use_gpu)
    else:
        use_gpu=False

    #only use pytorch randomness for direct usage with pytorch
    #check for pitfalls when using other modules
    random_seed = cfg['random_seed']
    if random_seed != None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(random_seed)


    # Parameters from config file, client training
    nnIsTrained = cfg['pre_trained']     # pre-trained using ImageNet
    trBatchSize = cfg['batch_size']

    # Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = cfg['imgtransResize']
    # imgtransCrop = cfg['imgtransCrop']
    policy = cfg['policy']

    class_idx = cfg['class_idx'] #indices of classes used for classification
    nnClassCount = len(class_idx)       # dimension of the output


    #federated learning parameters
    num_clients = cfg['num_clients']
    client_dirs = cfg['client_dirs']
    assert num_clients == len(client_dirs), "Number of clients doesn't correspond to number of directories specified"

    data_path = check_path(args.chexpert_path, warn_exists=False, require_exists=True)

    #define mean and std dependent on whether using a pretrained model
    if nnIsTrained:
        data_mean = IMAGENET_MEAN
        data_std = IMAGENET_STD
    else:
        data_mean = CHEXPERT_MEAN
        data_std = CHEXPERT_STD

    train_transformSequence = transforms.Compose([transforms.Resize((imgtransResize,imgtransResize)),
                                            # transforms.RandomResizedCrop(imgtransResize),
                                            # transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(data_mean, data_std)
                                            ])

    test_transformSequence = transforms.Compose([transforms.Resize((imgtransResize,imgtransResize)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(data_mean, data_std)
                                            ])

    #initialize client instances
    clients = [Client(name=f'client{n}') for n in range(num_clients)]
    for i in range(num_clients):
        cur_client = clients[i]
        print(f"Initializing {cur_client.name}")

        path_to_client = check_path(data_path + 'CheXpert-v1.0-small/' + client_dirs[i], warn_exists=False, require_exists=True)

        cur_client.train_file = path_to_client + 'client_train.csv'
        cur_client.val_file = path_to_client + 'client_train.csv'
        cur_client.test_file = path_to_client + 'client_test.csv'

        cur_client.train_data = CheXpertDataSet(data_path, cur_client.train_file, class_idx, policy, transform = train_transformSequence)
        cur_client.val_data = CheXpertDataSet(data_path, cur_client.val_file, class_idx, policy, transform = test_transformSequence)
        cur_client.test_data = CheXpertDataSet(data_path, cur_client.test_file, class_idx, policy, transform = test_transformSequence)

        assert cur_client.train_data[0][0].shape == torch.Size([3,imgtransResize,imgtransResize])
        assert cur_client.train_data[0][1].shape == torch.Size([nnClassCount])

        cur_client.n_data = cur_client.get_data_len()
        print(f"Holds {cur_client.n_data} data points")

        cur_client.train_loader = DataLoader(dataset=cur_client.train_data, batch_size=trBatchSize, shuffle=True,
                                            num_workers=4, pin_memory=True)
        # assert cur_client.train_loader.dataset == cur_client.train_data

        cur_client.val_loader = DataLoader(dataset=cur_client.val_data, batch_size=trBatchSize, shuffle=True,
                                            num_workers=4, pin_memory=True)
        cur_client.test_loader = DataLoader(dataset = cur_client.test_data, num_workers = 4, pin_memory = True)


    #create model
    if use_gpu:
        model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        model=torch.nn.DataParallel(model).cuda()
    else:
        model = DenseNet121(nnClassCount, nnIsTrained)

    # collect output per client
    outGT, outPRED, auroc_mean_clients = [], [], []

    for cl in clients:
        cl_outGT, cl_outPRED, cl_auroc_mean = Trainer.test(model, cl.test_loader, class_idx, use_gpu,
                                            checkpoint= args.model_path)
        outGT.append(cl_outGT)
        outPRED.append(cl_outPRED)
        auroc_mean_clients.append(cl_auroc_mean)

    # average AUC over all clients
    auroc_mean = np.nanmean(np.array(auroc_mean_clients))

    # print(outGT)
    # print(outPRED)
    print(auroc_mean)


def check_gpu_usage(use_gpu):

    """Give feedback to whether GPU is available and if the expected number of GPUs are visible to PyTorch.
    """
    assert use_gpu is True, "GPU not used"
    assert torch.cuda.device_count() == len(selected_gpus), "Wrong number of GPUs available to Pytorch"
    print(f"{torch.cuda.device_count()} GPUs available")

    return True




if __name__ == "__main__":
    main()
