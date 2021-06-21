
"""Train a model using federated learning"""

#set which GPUs to use
import os
selected_gpus = [6,7] #configure this
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in selected_gpus])

import pandas as pd
import argparse
import json
from PIL import Image
import time
import random

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

#local imports
from chexpert_data import CheXpertDataSet
from trainer import Trainer, DenseNet121
from utils import check_path


IMAGENET_MEAN = [0.485, 0.456, 0.406]  # mean of ImageNet dataset(for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]   # std of ImageNet dataset(for normalization)

CHEXPERT_MEAN = [0,0,0]
CHEXPERT_STD = [0,0,0] #TO DO compute


def main():
    use_gpu = torch.cuda.is_available()

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
    trMaxEpoch = cfg['max_epochs']

    # Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = cfg['imgtransResize']
    # imgtransCrop = cfg['imgtransCrop']
    policy = cfg['policy']

    class_idx = cfg['class_idx'] #indices of classes used for classification
    nnClassCount = len(class_idx)       # dimension of the output


    #federated learning parameters
    num_clients = cfg['num_clients']
    fraction = cfg['fraction']
    com_rounds = cfg['com_rounds']

    #run preprocessing to obtain these files
    data_path = check_path(args.chexpert_path, warn_exists=False, require_exists=True)
    pathFileTrain = data_path + 'CheXpert-v1.0-small/train_mod.csv'
    pathFileValid =  data_path + 'CheXpert-v1.0-small/valid_mod.csv'
    pathFileTest = data_path + 'CheXpert-v1.0-small/test_mod.csv'

    #define mean and std dependent on whether using a pretrained model
    if nnIsTrained:
        data_mean = IMAGENET_MEAN
        data_std = IMAGENET_STD
    else:
        data_mean = CHEXPERT_MEAN
        data_std = CHEXPERT_STD

    # define transforms
    # if using augmentation, use different transforms for training, test & val data
    transformSequence = transforms.Compose([transforms.Resize((imgtransResize,imgtransResize)),
                                            # transforms.RandomResizedCrop(imgtransResize),
                                            # transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            # transforms.Normalize(data_mean, data_std)
                                            ])

    # Load dataset
    datasetTrain = CheXpertDataSet(data_path, pathFileTrain, class_idx, policy, transform = transformSequence)
    len_train = len(datasetTrain)
    print("Train data length:", len_train)

    #remove transformations here?
    datasetValid = CheXpertDataSet(data_path, pathFileValid, class_idx, policy, transform = transformSequence)
    print("Valid data length:", len(datasetValid))

    datasetTest = CheXpertDataSet(data_path, pathFileTest, class_idx, policy, transform = transformSequence)
    print("Test data length:", len(datasetTest))

    assert datasetTrain[0][0].shape == torch.Size([3,imgtransResize,imgtransResize])
    assert datasetTrain[0][1].shape == torch.Size([nnClassCount])


    #IID distributed data, balanced, mixing patients between sites
    split_trainData = get_client_data_split(len_train, num_clients)

    #datasets and dataloaders for training data
    datasetsClients = random_split(datasetTrain, split_trainData)
    dataloadersClients = []

    for client_dataset in datasetsClients:
        dataloadersClients.append(DataLoader(dataset=client_dataset, batch_size=trBatchSize, shuffle=True,
                                            num_workers=4, pin_memory=True))

    #Create dataLoaders, normal training
    # dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True, num_workers=2, pin_memory=True)
    # print('Length train dataloader (n batches): ', len(dataLoaderTrain))

    #separate dataloaders for validation and testing
    dataLoaderVal = DataLoader(dataset = datasetValid, batch_size = trBatchSize, num_workers = 4, pin_memory = True)
    dataLoaderTest = DataLoader(dataset = datasetTest, num_workers = 4, pin_memory = True)



    #show images for testing
    # for batch in dataloadersClients[0]:
    #     # transforms.ToPILImage()(batch[0][0]).show()
    #     print(batch[1][0])


    #create model
    if use_gpu:
        model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        model=torch.nn.DataParallel(model).cuda()
    else:
        model = DenseNet121(nnClassCount, nnIsTrained)

    #define path to store results in
    output_path = check_path(args.output_path, warn_exists=True)

    fed_start = time.time()
    #FEDERATED LEARNING
    for i in range(com_rounds):

        print(f"[[[ Round {i} Start ]]]")
        client_params = [None] * num_clients #store all client models

        # Step 1: select random fraction of clients
        if fraction < 1:
            sel_clients = sorted(random.sample(range(num_clients),
                                           round(num_clients*fraction)))
        else:
            sel_clients = list(range(num_clients))
        print("Number of selected clients: ", len(sel_clients))
        print("Clients selected: ", sel_clients)

        # Step 2: send global model to clients and train locally
        for client_k in sel_clients:

            print(f"<< Client {client_k} Training Start >>")
            client_output_path = output_path + f"round{i}_client{client_k}/"
            client_output_path = check_path(client_output_path, warn_exists=False)
            print(client_output_path)

            train_valid_start = time.time()
            client_params[client_k] = Trainer.train(model, dataloadersClients[client_k], dataLoaderVal, # Step 3: Perform local computations
                                              class_idx, cfg, client_output_path, use_gpu, out_csv=f"round{i}_client{client_k}.csv")

            train_valid_end = time.time()
            client_time = round(train_valid_end - train_valid_start)
            print(f"<< Client {client_k} Training Time: {client_time} seconds >>")

        trained_clients = [idx for idx in range(num_clients) if client_params[idx] != None]
        first_idx = trained_clients[0]
        last_idx = trained_clients[-1]
        print(trained_clients)
        # Step 4: return updates to server
        for key in client_params[first_idx]: #iterate through parameters layerwise
            weights, weightn = [], []

            for clie in sel_clients:
                weights.append(client_params[clie][key]*split_trainData[clie])
                weightn.append(split_trainData[clie])
            #store in parameter list, last client
            client_params[last_idx][key] = sum(weights) / sum(weightn) # weighted averaging model weights

        if use_gpu:
            model = DenseNet121(nnClassCount).cuda()
            model = torch.nn.DataParallel(model).cuda()
        # Step 5: server updates global state
        model.load_state_dict(client_params[last_idx])
        print(f"[[[ Round {i} End ]]]\n")

    print("Global model trained")
    fed_end = time.time()
    print(f"Total training time: {round(fed_end-fed_start,0)}")

    #save for inference
    torch.save(model.state_dict(), output_path+f"global_{com_rounds}rounds.pth.tar")

    #normal training
    # start = time.time()
    # params = Trainer.train(model, dataLoaderTrain, dataLoaderVal, class_idx, cfg, output_path, use_gpu)
    # end = time.time()
    # print(f"Total time: {end-start}")

    # outGT, outPRED, auroc_mean = Trainer.test(model, dataLoaderTest, class_idx, use_gpu,
    #                                     checkpoint= output_path+f"global_{com_rounds}rounds.pth.tar")



def check_gpu_usage(use_gpu):

    """Gives feedback to whether GPU is available and if the expected number of GPUs are visible to PyTorch.
    """
    assert use_gpu is True, "GPU not used"
    assert torch.cuda.device_count() == len(selected_gpus), "Wrong number of GPUs available to Pytorch"
    print(f"{torch.cuda.device_count()} GPUs available")

    return True

def get_client_data_split(len_data, num_clients):

    """Returns a list with amount of data that should be provided to clients.
    One list element represents one client.
    For now assumes that data should be balanced between clients.
    """

    print(f"{num_clients} clients")
    data_per_client = len_data//num_clients
    print(f"Data per client: ", data_per_client)
    data_left = len_data - data_per_client*num_clients
    print(f"Data left: ", data_left)
    #last client gets the rest, will be at most num_clients different from others
    data_split = [data_per_client for i in range(num_clients-1)] + [data_per_client + data_left]
    print(f"Data-client split: ", data_split)

    return data_split




if __name__ == "__main__":
    main()
