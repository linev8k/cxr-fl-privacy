"""Train a model using federated learning"""

#set which GPUs to use
import os
selected_gpus = [0] #configure this
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
from utils import check_path, merge_eval_csv


IMAGENET_MEAN = [0.485, 0.456, 0.406]  # mean of ImageNet dataset(for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]   # std of ImageNet dataset(for normalization)

CHEXPERT_MEAN = [0.5029, 0.5029, 0.5029]
CHEXPERT_STD = [0.2899, 0.2899, 0.2899]


def main():
    use_gpu = torch.cuda.is_available()

    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #parse config file
    parser.add_argument('cfg_path', type = str, help = 'Path to the config file in json format.')
    #output path for storing results
    parser.add_argument('--output_path', '-o', help = 'Path to save results.', default = 'results/')
    #whether to assert GPU usage (disable for testing without GPU)
    parser.add_argument('--no_gpu', dest='no_gpu', help='Don\'t verify GPU usage.', action='store_true')
    #set path to data (Chexpert and Mendeley)
    parser.add_argument('--data', '-d', dest='data_path', help='Path to data.', default='./')
    #specify path to client files for data reading
    parser.add_argument('--data_files', '-df', dest='data_files', help='Path to data files.', default='./')
    parser.add_argument('--model', '-m', dest='model', help='Model to load weights from.', default=None)
    parser.add_argument('--combine', dest='combine', help="Combine CheXpert and Mendeley data", action='store_true')

    args = parser.parse_args()
    with open(args.cfg_path) as f:
        cfg = json.load(f)

    if not args.no_gpu:
        check_gpu_usage(use_gpu)
    else:
        use_gpu=False

    if args.combine: # adjust this manually if needed
        print("Combining CheXpert and Mendeley clients")
        chexpert_client_n = list(range(20,25))
        mendeley_client_n = list(range(0,20))
        assert cfg['num_clients'] == len(chexpert_client_n)+len(mendeley_client_n), "Check client combination"


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
    freeze_mode = cfg['freeze'] # what layers to freeze: 'none', 'batch_norm'
    trBatchSize = cfg['batch_size']
    trMaxEpoch = cfg['max_epochs']

    if cfg['net'] == 'DenseNet121':
        net = DenseNet121
    elif cfg['net'] == 'ResNet50':
        net = ResNet50
    model_checkpoint = args.model

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
    client_dirs = cfg['client_dirs']
    assert num_clients == len(client_dirs), "Number of clients doesn't correspond to number of directories specified"
    fraction = cfg['fraction']
    com_rounds = cfg['com_rounds']
    earl_stop_rounds = cfg['earl_stop_rounds']
    reduce_lr_rounds = cfg['reduce_lr_rounds']


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
    # if using augmentation, use different transforms for training, test & val data
    if augment:
        train_transformSequence = transforms.Compose([transforms.Resize((imgtransResize,imgtransResize)),
                                                # transforms.RandomResizedCrop(imgtransResize),
                                                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(data_mean, data_std)
                                                ])
    else:
        #no augmentation for comparison with DP
        train_transformSequence = transforms.Compose([transforms.Resize((imgtransResize,imgtransResize)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(data_mean, data_std)
                                                ])

    test_transformSequence = transforms.Compose([transforms.Resize((imgtransResize,imgtransResize)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(data_mean, data_std)
                                            ])

    #initialize client instances
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
        print(path_to_client)

        cur_client.train_file = path_to_client + 'client_train.csv'
        cur_client.train_data = CheXpertDataSet(data_path, cur_client.train_file, class_idx, policy, colour_input=colour_input, transform = train_transformSequence)

        assert cur_client.train_data[0][0].shape == torch.Size([len(colour_input),imgtransResize,imgtransResize])
        assert cur_client.train_data[0][1].shape == torch.Size([nnClassCount])

        cur_client.n_data = cur_client.get_data_len()
        print(f"Holds {cur_client.n_data} data points")

        cur_client.train_loader = DataLoader(dataset=cur_client.train_data, batch_size=trBatchSize, shuffle=True,
                                            num_workers=4, pin_memory=True)
        # assert cur_client.train_loader.dataset == cur_client.train_data

        cur_client.val_file = path_to_client + 'client_val.csv'
        cur_client.test_file = path_to_client + 'client_test.csv'

        if os.path.exists(cur_client.val_file):
            cur_client.val_data = CheXpertDataSet(data_path, cur_client.val_file, class_idx, policy, colour_input=colour_input, transform = test_transformSequence)
            cur_client.test_data = CheXpertDataSet(data_path, cur_client.test_file, class_idx, policy, colour_input=colour_input, transform = test_transformSequence)

            cur_client.val_loader = DataLoader(dataset=cur_client.val_data, batch_size=trBatchSize, shuffle=False,
                                                num_workers=4, pin_memory=True)
            cur_client.test_loader = DataLoader(dataset = cur_client.test_data, num_workers = 4, pin_memory = True)

        else: # clients that don't
            print(f"No validation data for client{i}")
            cur_client.val_loader = None
            cur_client.test_loader = None

    # show images for testing
   #  for batch in clients[1].train_loader:
    #     transforms.ToPILImage()(batch[0][0]).show()
     #    print(batch[1])
    #
    #  for batch in clients[1].val_loader:
    #     transforms.ToPILImage()(batch[0][0]).show()
     #     print(batch[1][0])

     # get labels and indices of data from dataloaders
     # modify chexpert_data dataloader to also return indices for this
     # for i in [12,13,14,15,16,17,18,19]:
     #     print("Client ", i)
     #     for batch in clients[i].train_loader:
     #         print("Labels: ", batch[1])
     #         print("Inidces: ", batch[2])


    #create model
    if use_gpu:
        global_model = net(nnClassCount, colour_input, nnIsTrained).cuda()
        # model=torch.nn.DataParallel(model).cuda()
    else:
        global_model = net(nnClassCount, colour_input, nnIsTrained)

    if model_checkpoint is not None: # load weights if some model is specified
        checkpoint = torch.load(model_checkpoint)
        if 'state_dict' in checkpoint:
            global_model.load_state_dict(checkpoint['state_dict'])
        else:
            global_model.load_state_dict(checkpoint)

    #define path to store results in
    output_path = check_path(args.output_path, warn_exists=True)

    torch.save({'state_dict': global_model.state_dict()}, output_path+'global_init.pth.tar')

    fed_start = time.time()
    #FEDERATED LEARNING
    global_auc = []
    best_global_auc = 0
    track_no_improv = 0

    for i in range(com_rounds):

        print(f"[[[ Round {i} Start ]]]")

        # Step 1: select random fraction of clients
        if fraction < 1:
            sel_clients = sorted(random.sample(clients,
                                           round(num_clients*fraction)))
        else:
            sel_clients = clients
        print("Number of selected clients: ", len(sel_clients))
        print(f"Clients selected: {[sel_cl.name for sel_cl in sel_clients]}")
        # Step 2: send global model to clients and train locally
        for client_k in sel_clients:

            # reset model at client's site
            client_k.model_params = None

            # create independent copy of initial model with respective parameters
            if use_gpu:
                local_model = net(nnClassCount, colour_input, pre_trained=False).cuda()
            else:
                local_model = net(nnClassCount, colour_input, pre_trained=False)
            local_model.load_state_dict(global_model.state_dict())

            print(f"<< {client_k.name} Training Start >>")
            # set output path for storing models and results
            client_k.output_path = output_path + f"round{i}_{client_k.name}/"
            client_k.output_path = check_path(client_k.output_path, warn_exists=False)
            print(client_k.output_path)

            train_valid_start = time.time()
            # Step 3: Perform local computations
            # returns local best model
            client_k.model_params = Trainer.train(local_model, client_k.train_loader, client_k.val_loader,
                                               cfg, client_k.output_path, use_gpu, out_csv=f"round{i}_{client_k.name}.csv",
                                               freeze_mode = freeze_mode)
            train_valid_end = time.time()
            client_time = round(train_valid_end - train_valid_start)
            print(f"<< {client_k.name} Training Time: {client_time} seconds >>")

        first_cl = sel_clients[0]
        # Step 4: return updates to server
        for key in first_cl.model_params: #iterate through parameters layerwise
            weights, weightn = [], []

            for cl in sel_clients:
                weights.append(cl.model_params[key]*len(cl.train_data))
                weightn.append(len(cl.train_data))

            #store parameters with first client for convenience
            first_cl.model_params[key] = sum(weights) / sum(weightn) # weighted averaging model weights

        if use_gpu:
           global_model = net(nnClassCount, colour_input).cuda()
           # model = torch.nn.DataParallel(model).cuda()
        # Step 5: server updates global state
        global_model.load_state_dict(first_cl.model_params)

        # also save intermediate models
        torch.save(global_model.state_dict(),
                   output_path + f"global_{i}rounds.pth.tar")

       #validate global model on client validation data
        print("Validating global model...")
        aurocMean_global = []
        for cl in clients:
            if cl.val_loader is not None:
                GT, PRED, cl_aurocMean, _ = Trainer.test(global_model, cl.val_loader, class_idx, use_gpu, checkpoint=None)
                aurocMean_global.append(cl_aurocMean)
            else:
                aurocMean_global.append(np.nan)
           #  print(GT)
           # print(PRED)
        cur_global_auc = np.nanmean(np.array(aurocMean_global))
        print("AUC Mean: {:.3f}".format(cur_global_auc))
        global_auc.append(cur_global_auc)

        # track early stopping & lr decay
        if cur_global_auc > best_global_auc:
            best_global_auc = cur_global_auc
            track_no_improv = 0
        else:
            track_no_improv += 1
            if track_no_improv == reduce_lr_rounds:
                # decay lr
                cfg['lr'] = cfg['lr'] * 0.1
                print(f"Learning rate reduced to {cfg['lr']}")
            elif track_no_improv == earl_stop_rounds:
                print(f'Global AUC has not improved for {earl_stop_rounds} rounds. Stopping training.')
                break

        print(f"[[[ Round {i} End ]]]\n")

    # save global AUC in CSV
    all_metrics = [list(range(com_rounds)), global_auc]
    with open(output_path+'global_validation.csv', 'w') as f:
        header = ['round', 'val AUC']
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(zip(*all_metrics))

    print("Global model trained")
    fed_end = time.time()
    print(f"Total training time: {round(fed_end-fed_start,0)}")

    # merge local metrics to CSV
    try:
        merge_eval_csv(output_path, out_file='train_results.csv')
    except:
        print("Merging result CSVs failed. Try again manually.")


def check_gpu_usage(use_gpu):

    """Give feedback to whether GPU is available and if the expected number of GPUs are visible to PyTorch.
    """
    assert use_gpu is True, "GPU not used"
    assert torch.cuda.device_count() == len(selected_gpus), "Wrong number of GPUs available to Pytorch"
    print(f"{torch.cuda.device_count()} GPUs available")

    return True




if __name__ == "__main__":
    main()
