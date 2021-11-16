"""Module for training.
Originally based on and modified from https://github.com/Stomper10/CheXpert/blob/master/materials.py.

Contains:
    A Trainer class which bundles relevant functions for training, validation, testing.
    Custom modules for DenseNet121 and ResNet50.
    Functions for layer freezing.
    A Client class for managing individual FL clients.
"""

import time
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import csv

import torch
import torchvision
from torch import nn
from torch import optim
from torch.backends import cudnn

class Trainer():

    def train(client_k, cfg, use_gpu, out_csv='train_log.csv', checkpoint=None, freeze_mode='none'):

        """Train a local model from a client instance.
        Args:
            client_k (Client object): Client instance with client model, data laoders, output path attributes.
            cfg (dict): Config dictionary containing training parameters.
            use_gpu (bool): Whether to use available GPUs.
            out_csv (str): Name of CSV file used for logging. Stored in output path.
            checkpoint (str): A model checkpoint to load from for continuing training.
            freeze_mode (str): Information about which layers to freeze during training.
        Returns nothing.
        """
        out_csv_path = client_k.output_path + out_csv

        loss = torch.nn.BCELoss() # setting binary cross entropy as loss function

        if checkpoint != None: # load checkpoint
            modelCheckpoint = torch.load(checkpoint)
            client_k.model.load_state_dict(modelCheckpoint['state_dict'])
            client_k.optimizer.load_state_dict(modelCheckpoint['optimizer'])
        params = client_k.model.state_dict().copy()

        # logging metrics
        lossMIN = 100000
        train_start = []
        train_end = []

        save_epoch = []
        save_train_loss = []
        save_val_loss = []
        save_val_AUC = []
        save_epsilon = []
        save_alpha = []
        save_delta = []

        # train model for number of epochs
        for epochID in range(0, cfg['max_epochs']):
            train_start.append(time.time())
            losst = Trainer.epochTrain(client_k.model, client_k.train_loader, client_k.optimizer, loss, use_gpu, freeze_mode=freeze_mode)
            train_end.append(time.time())

            # model validation
            if client_k.val_loader is not None:
                print("Validating model...")
                lossv, aurocMean = Trainer.epochVal(client_k.model, client_k.val_loader, loss, use_gpu)
                print("Training loss: {:.3f},".format(losst), "Valid loss: {:.3f}".format(lossv))
            else:
                # if the client doesn't have validation data, add nan placeholders to metrics
                lossv, aurocMean = (np.nan, np.nan)
                # store model parameters regardless of validation
                params = client_k.model.state_dict().copy()

            # save model to intermediate checkpoint file
            model_num = epochID + 1
            torch.save({'epoch': model_num, 'state_dict': client_k.model.state_dict(),
                        'loss': lossMIN, 'optimizer' : client_k.optimizer.state_dict()},
                       f"{client_k.output_path}{model_num}-epoch_FL.pth.tar")

            # keep parameters of best model
            if lossv < lossMIN:
                lossMIN = lossv
                print('Epoch ' + str(model_num) + ' [++++] val loss decreased')
                params = client_k.model.state_dict().copy()
            else:
                print('Epoch ' + str(model_num) + ' [----] val loss did not decrease or no val data available')

            # track metrics
            save_epoch.append(model_num)
            save_train_loss.append(losst)
            save_val_loss.append(lossv)
            save_val_AUC.append(aurocMean)

            if cfg['private']:
                epsilon, best_alpha = client_k.optimizer.privacy_engine.get_privacy_spent()
                print(f"epsilon: {epsilon:.2f}, best alpha: {best_alpha}")
                save_epsilon.append(epsilon)
                save_alpha.append(best_alpha)
                save_delta.append(client_k.delta)

            if cfg['track_norm']:
                # follow L2 grad norm per parameter layer
                grad_norm = []
                for p in list(filter(lambda p: p.grad is not None, client_k.model.parameters())):
                    cur_norm = p.grad.data.norm(2).item()
                    grad_norm.append(cur_norm)

        train_time = np.array(train_end) - np.array(train_start)
        print("Training time for each epoch: {} seconds".format(train_time.round(0)))

        # save logging metrics in CSV
        all_metrics = [save_epoch, train_time, save_train_loss, save_val_loss, save_val_AUC]
        if cfg['track_norm']:
            all_metrics += [[grad_norm]]
        if cfg['private']:
            all_metrics += [save_epsilon, save_alpha, save_delta]

        with open(out_csv_path, 'w') as f:
            header = ['epoch', 'time', 'train loss', 'val loss', 'val AUC']
            if cfg['track_norm']:
                header += ['track_norm']
            if cfg['private']:
                header += ['epsilon', 'best_alpha', 'delta']
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(zip(*all_metrics))

        return

    def epochTrain(model, dataLoaderTrain, optimizer, loss, use_gpu, freeze_mode='none'):

        """Train a model for one epoch.
        Args:
            model (model object): Model to train.
            dataLoaderTrain (dataloader object): PyTorch ataloader with training data.
            optimizer (optimizer object): Optimizer instance.
            loss (function object): Loss function.
            use_gpu (bool): Whether to train on GPU.
            freeze_mode (str): Information about which layers to freeze.
        Returns:
            (float) Mean training loss over batches."""

        losstrain = 0
        model.train()

        if freeze_mode == 'batch_norm':
            freeze_batchnorm(model)
        if freeze_mode == 'all_but_last':
            freeze_all_but_last(model)

        # usual training procedure
        with tqdm(dataLoaderTrain, unit='batch') as tqdm_loader:

            for varInput, target in tqdm_loader:

                if use_gpu:
                    target = target.cuda(non_blocking = True)
                    varInput = varInput.cuda(non_blocking=True)

                varOutput = model(varInput) #forward pass
                lossvalue = loss(varOutput, target)

                optimizer.zero_grad() #reset gradient
                lossvalue.backward()
                optimizer.step()

                losstrain += lossvalue.item()

                tqdm_loader.set_postfix(loss=lossvalue.item())

        return losstrain / len(dataLoaderTrain)


    def epochVal(model, dataLoaderVal, loss, use_gpu):

        """Validate a model.
        Args:
            model (model object): Model to validate.
            dataLoaderVal (dataloader object): PyTorch ataloader with validation data.
            loss (function object): Loss function.
            use_gpu (bool): Whether to train on GPU.
        Returns:
            (float): Mean validation loss over batches.
            (float): Mean AUROC over all labels."""

        model.eval()
        lossVal = 0

        if use_gpu:
            outGT = torch.FloatTensor().cuda()
            outPRED = torch.FloatTensor().cuda()
        else:
            outGT = torch.FloatTensor()
            outPRED = torch.FloatTensor()

        with torch.no_grad():
            for varInput, target in dataLoaderVal:

                if use_gpu:
                    target = target.cuda(non_blocking = True)
                    varInput = varInput.cuda(non_blocking=True)

                varOutput = model(varInput)

                lossVal += loss(varOutput, target).item()

                # collect predictions and ground truth for AUROC computation
                outGT = torch.cat((outGT, target), 0)
                outPRED = torch.cat((outPRED, varOutput), 0)

        # compute AUROC mean
        aurocIndividual = Trainer.computeAUROC(outGT, outPRED)
        aurocMean = np.nanmean(np.array(aurocIndividual))
        print('AUROC mean: {:.4f}'.format(aurocMean))

        return lossVal / len(dataLoaderVal), aurocMean


    def computeAUROC(dataGT, dataPRED):

        """Compute Area under Receiver Operating Characteristic curve.
        Args:
            dataGT (tensor): Ground truth labels.
            dataPRED (tensor): Predicted labels.
        Returns:
            (list): AUROC for each label."""

        outAUROC = []
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        nnClassCount = dataGT.shape[1] # [0] is the batch size

        for i in range(nnClassCount):
            try:
                outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            # AUROC is not defined for a single data point
            # or there are labels which are negative in all data points
            except ValueError:
                print(f"AUROC not defined for label {i}")
                outAUROC.append(np.nan)

        return outAUROC


    def test(model, dataLoaderTest, class_idx, use_gpu, checkpoint=None):

        """Stand-alone function for testing a model.
        Args:
            model (model object): Model to validate.
            dataLoaderTest (dataloader object): PyTorch ataloader with test data.
            class_idx (list): List of label indices with which the model has been trained.
            use_gpu (bool): Whether to train on GPU.
            checkpoint (str): A model checkpoint to load parameters from.
        Returns:
            (tensor): Ground truth labels.
            (tensor): Predicted labels.
            (float) Mean AUROC over all labels.
            (list): Individual AUROC for each label."""

        model.eval()
        nnClassCount = len(class_idx)
        class_names = dataLoaderTest.dataset.class_names

        if use_gpu:
            outGT = torch.FloatTensor().cuda()
            outPRED = torch.FloatTensor().cuda()
        else:
            outGT = torch.FloatTensor()
            outPRED = torch.FloatTensor()

        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            if 'state_dict' in modelCheckpoint:
                model.load_state_dict(modelCheckpoint['state_dict'])
            else:
                model.load_state_dict(modelCheckpoint)

        with torch.no_grad():
            for i, (varInput, target) in enumerate(dataLoaderTest):

                if use_gpu:
                    target = target.cuda()
                    varInput = varInput.cuda()
                outGT = torch.cat((outGT, target), 0)

                out = model(varInput)
                outPRED = torch.cat((outPRED, out), 0)

        aurocIndividual = Trainer.computeAUROC(outGT, outPRED)
        aurocMean = np.nanmean(np.array(aurocIndividual))
        print('AUROC mean: {:.4f}  '.format(aurocMean))

        for i in range(0, len(aurocIndividual)):
            print(class_names[class_idx[i]], ': {:.4f}  '.format(aurocIndividual[i]))

        return outGT, outPRED, aurocMean, aurocIndividual

class DenseNet121(nn.Module):

    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size, colour_input='RGB', pre_trained=False):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained = pre_trained)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

        if colour_input == 'L':
            self.rgb_to_grey_input()

    def forward(self, x):
        x = self.densenet121(x)
        return x

    def rgb_to_grey_input(self):

        """Replace the first convolutional layer that takes a 3-dimensional (RGB) input
        with a 1-dimensional layer, adding the weights of each existing dimension
        in order to retain pretrained parameters"""

        conv0_weight = self.densenet121.features.conv0.weight.clone()
        self.densenet121.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            self.densenet121.features.conv0.weight = nn.Parameter(conv0_weight.sum(dim=1,keepdim=True)) # way to keep pretrained weights

    def get_n_params(self, trainable=True):
        """Return number of (trainable) parameters."""

        if trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

class ResNet50(nn.Module):

    """Model modified.
    The architecture of our model is the same as standard ResNet18
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self, out_size, colour_input = 'RGB', pre_trained=False):
        super(ResNet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained = pre_trained)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

        if colour_input == 'L':
            self.rgb_to_grey_input()

    def forward(self, x):
        x = self.resnet50(x)
        return x

    def rgb_to_grey_input(self):

        """Replace the first convolutional layer that takes a 3-dimensional (RGB) input
        with a 1-dimensional layer, adding the weights of each existing dimension
        in order to retain pretrained parameters"""

        conv1_weight = self.resnet50.conv1.weight.clone()
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            self.resnet50.conv1.weight = nn.Parameter(conv1_weight.sum(dim=1,keepdim=True)) # way to keep pretrained weights

    def get_n_params(self, trainable=True):
        """Return number of (trainable) parameters."""

        if trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

def freeze_batchnorm(model):

    """Modify model to not track gradients of batch norm layers
    and set them to eval() mode (no running stats updated)"""

    for module in model.modules():
        # print(module)
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

def freeze_all_but_last(model):

    """Modify model to not track gradients of all but the last classification layer.
    Note: This is customized to the module naming of ResNet and DenseNet architectures."""

    for name, param in model.named_parameters():
        if 'fc' not in name and 'classifier' not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

class Client():

    """Class for instantiating a single client.
    Mainly for tracking relevant objects during the course of training for each client.
    Init args:
        name (str): Name of client."""

    def __init__(self, name):

        """Placeholders for attributes."""

        self.name = name

        # datasets and loaders
        # dataloaders track changes in associated datasets
        # so we need to uniquely associate constant datasets with the client
        self.train_data = None
        self.train_loader = None
        self.val_data = None
        self.val_loader = None
        self.test_data = None
        self.test_loader = None

        self.n_data = None # size of training dataset
        self.output_path = None # name of output path for storing results

        # local model objects
        self.model_params = None # state dict of model
        self.model = None
        self.optimizer = None

        # individual privacy objects/parameters
        self.privacy_engine = None
        self.delta = None
        self.grad_norm = []

    def init_optimizer(self, cfg):

        """Initialize client optimizer and set it as client attribute.
        Args:
            cfg (dict): Training config dictionary with optimizer information.
        Returns nothing."""

        if self.model != None:
            if cfg['optim'] == "Adam":
                self.optimizer = optim.Adam(self.model.parameters(), lr = cfg['lr'], # setting optimizer & scheduler
                                       betas = tuple(cfg['betas']), eps = cfg['eps'], weight_decay = cfg['weight_decay'])
            if cfg['optim'] == "SGD":
                self.optimizer = optim.SGD(self.model.parameters(), lr=cfg['lr'])
        else:
            print("self.model is currently None. Optimizer cannot be initialized.")

    def get_data_len(self):

        """Return number of data points (int) currently held by client, all splits taken together."""

        n_data = 0
        for data in [self.train_data, self.val_data, self.test_data]:
            if data != None:
                n_data += len(data)

        return n_data
