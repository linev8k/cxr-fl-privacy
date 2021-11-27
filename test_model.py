
"""Validate a model on each client's validation or test data."""

#set which GPUs to use
import os
selected_gpus = [1] #configure this
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in selected_gpus])

import pandas as pd
import argparse
import json
from PIL import Image
import time
import random
import numpy as np
import csv

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#local imports
from chexpert_data import CheXpertDataSet
from trainer import Trainer, DenseNet121, ResNet50, Client
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
    #model checkpoint path
    parser.add_argument('--model', '-m', dest='model_path', help='Path to model.', required=True)
    #output path for storing results
    parser.add_argument('--output_path', '-o', help = 'Path to save results.', default = 'results/')
    #output file for storing results
    parser.add_argument('--output_file', '-of', help = 'CSV file for saving results.', default = 'auc.csv')
    #set path to chexpert data
    parser.add_argument('--data', '-d', dest='data_path', help='Path to data.', default='./')
    #specify path to client files for data reading
    parser.add_argument('--data_files', '-df', dest='data_files', help='Path to data files.', default='./')
    #whether to assert GPU usage (disable for testing without GPU)
    parser.add_argument('--no_gpu', dest='no_gpu', help='Don\'t verify GPU usage.', action='store_true')
    parser.add_argument('--val', dest='use_val', help='Whether to use validation data. Test data is used by default.', action='store_true')
    parser.add_argument('--combine', dest='combine', help="Combine CheXpert and Mendeley data.", action='store_true')


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

    if args.combine: # adjust this manually if needed
        print("Combining CheXpert and Mendeley clients")
        chexpert_client_n = list(range(14,36))
        mendeley_client_n = list(range(0,14))
        assert cfg['num_clients'] == len(chexpert_client_n)+len(mendeley_client_n), "Check client combination"


    # Parameters from config file, client training
    nnIsTrained = cfg['pre_trained']     # pre-trained using ImageNet
    trBatchSize = cfg['batch_size']
    trMaxEpoch = cfg['max_epochs']

    if cfg['net'] == 'DenseNet121':
        net = DenseNet121
    elif cfg['net'] == 'ResNet50':
        net = ResNet50

    # Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = cfg['imgtransResize']
    # imgtransCrop = cfg['imgtransCrop']
    policy = cfg['policy']
    colour_input = cfg['input']
    augment = cfg['augment']

    class_idx = cfg['class_idx'] #indices of classes used for classification
    nnClassCount = len(class_idx)       # dimension of the output


    #federated learning parameters
    num_clients = cfg['num_clients']
    client_dirs = [f"client{num}/" for num in range(num_clients)]
    fraction = cfg['fraction']
    com_rounds = cfg['com_rounds']

    data_path = check_path(args.data_path, warn_exists=False, require_exists=True)
    data_files = check_path(args.data_files, warn_exists=False, require_exists=True)

    #define mean and std dependent on whether using a pretrained model
    if nnIsTrained:
        data_mean = IMAGENET_MEAN
        data_std = IMAGENET_STD
    else:
        data_mean = CHEXPERT_MEAN
        data_std = CHEXPERT_STD
    if colour_input == 'L':
        data_mean = np.mean(data_mean)
        data_std = np.mean(data_std)

    # define transforms
    test_transformSequence = transforms.Compose([transforms.Resize((imgtransResize,imgtransResize)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(data_mean, data_std)
                                            ])

    num_no_val = 0
    #initialize client instances and their datasets
    data_files = check_path(args.data_files, warn_exists=False, require_exists=True)
    clients = [Client(name=f'client{n}') for n in range(num_clients)]
    for i in range(num_clients):

        cur_client = clients[i]
        print(f"Initializing {cur_client.name}")

        if args.combine:
            if i in chexpert_client_n:
                data_path = check_path(args.data_path+'ChestXrays/CheXpert/', warn_exists=False, require_exists=True)
            elif i in mendeley_client_n:
                data_path = check_path(args.data_path, warn_exists=False, require_exists=True)
        else:
            data_path = check_path(args.data_path, warn_exists=False, require_exists=True)

        path_to_client = check_path(data_files + client_dirs[i], warn_exists=False, require_exists=True)

        val_file = path_to_client + 'client_val.csv'
        test_file = path_to_client + 'client_test.csv'

        if os.path.exists(val_file):
            cur_client.val_data = CheXpertDataSet(data_path, val_file, class_idx, policy, colour_input=colour_input, transform = test_transformSequence)
            cur_client.test_data = CheXpertDataSet(data_path, test_file, class_idx, policy, colour_input=colour_input, transform = test_transformSequence)

            cur_client.val_loader = DataLoader(dataset=cur_client.val_data, batch_size=trBatchSize, shuffle=False,
                                                num_workers=4, pin_memory=True)
            cur_client.test_loader = DataLoader(dataset = cur_client.test_data, num_workers = 4, pin_memory = True)

        else: # clients that don't
            print(f"No validation data for client{i}")
            cur_client.val_loader = None
            cur_client.test_loader = None
            num_no_val += 1

    # show images for testing
    # for batch in clients[0].train_loader:
    #     transforms.ToPILImage()(batch[0][0]).show()
    #     print(batch[1][0])
    #
    # for batch in clients[0].val_loader:
    #     transforms.ToPILImage()(batch[0][0]).show()
    #     print(batch[1][0])


    # create model
    if use_gpu:
        model = net(nnClassCount, colour_input, pre_trained=False).cuda()
    else:
        model = net(nnClassCount, colour_input, pre_trained=False)

    # define path to store results in
    output_path = check_path(args.output_path, warn_exists=True)

    # read model checkpoint
    checkpoint = args.model_path

    #validate global model on client validation data
    print("Validating model on each client's data...")

    aurocMean_global_clients = [] # list of AUCs of clients
    aurocMean_individual_clients = [0 for idx in class_idx] # collect summed AUCs for each finding

    # check if validation or test data should be used
    if args.use_val:
        print('Using validation data')

    for cl in clients:
        if args.use_val:
            use_dataloader = cl.val_loader
        else:
            use_dataloader = cl.test_loader

        print(cl.name)
        if use_dataloader is not None:
            # get AUC mean and per class for current client
            LABEL, PRED, cl_aurocMean, aurocIndividual = Trainer.test(model, use_dataloader, class_idx, use_gpu, checkpoint=checkpoint)
            aurocMean_global_clients.append(cl_aurocMean)
            for i in range(nnClassCount):
                # track sum per class over all clients
                aurocMean_individual_clients[i] += aurocIndividual[i]
            # print(LABEL)
            # print(PRED)
        else:
            aurocMean_global_clients.append(np.nan)
            print(f"No data available for {cl.name}")

    # get mean of per class AUCs of all clients
    aurocMean_individual_clients = [auc/(num_clients-num_no_val) for auc in aurocMean_individual_clients]
    for i in range(nnClassCount):
        print(f'Mean for label {class_idx[i]}: {aurocMean_individual_clients[i]}  ')

    # overall mean of client AUCs
    auc_global = np.nanmean(np.array(aurocMean_global_clients))
    print("AUC Mean of all clients: {:.3f}".format(auc_global))
    aurocMean_global_clients.append(auc_global) # save global mean
    save_clients = [cl.name for cl in clients]
    save_clients.append('avg')

    # save AUC in CSV
    print(f'Saving in {output_path+args.output_file}')
    all_metrics = [save_clients, aurocMean_global_clients]
    with open(output_path+args.output_file, 'w') as f:
        header = ['client', 'AUC']
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(zip(*all_metrics))


def check_gpu_usage(use_gpu):

    """Give feedback to whether GPU is available and if the expected number of GPUs are visible to PyTorch.
    """
    assert use_gpu is True, "GPU not used"
    assert torch.cuda.device_count() == len(selected_gpus), "Wrong number of GPUs available to Pytorch"
    print(f"{torch.cuda.device_count()} GPUs available")

    return True




if __name__ == "__main__":
    main()
