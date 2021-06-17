"""Module for training"""

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

#TODO
#monitor training progress: loss, val accuracy (?)


class Trainer():

    def train(model, dataLoaderTrain, dataLoaderVal, class_idx, cfg, output_path, use_gpu, out_csv='train_log.csv', checkpoint=None):


        """Train a model.
        Args
            model: Instance of the model to be trained.
            dataLoaderTrain: Dataloader for training data.
            dataLoaderVal: Dataloader for validation data.
            class_idx: Indices of labels to use for training.
            cfg: Config dictionary containing training parameters.
            output_path: Path where results should be saved.
            use_gpu: Whether to use available GPUs.
            out_csv: Name of CSV file used for logging. Stored in output path.
            checkpoint: A model checkpoint to load from for continuing training.
        """
        out_csv_path = output_path + out_csv

        optimizer = optim.Adam(model.parameters(), lr = cfg['lr'], # setting optimizer & scheduler
                               betas = tuple(cfg['betas']), eps = cfg['eps'], weight_decay = cfg['weight_decay'])
        loss = torch.nn.BCELoss() # setting binary cross entropy as loss function

        if checkpoint != None: # loading checkpoint
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])

        # Train the network
        lossMIN = 100000
        train_start = []
        train_end = []

        #logging metrics
        save_epoch = []
        save_train_loss = []
        save_val_loss = []
        save_val_AUC = []

        for epochID in range(0, cfg['max_epochs']):
            train_start.append(time.time()) # training starts
            losst = Trainer.epochTrain(model, dataLoaderTrain, optimizer, loss, use_gpu)
            train_end.append(time.time()) # training ends

            #validation
            print("Validating model...")
            lossv, aurocMean = Trainer.epochVal(model, dataLoaderVal, optimizer, loss, use_gpu)
            print("Training loss: {:.3f},".format(losst), "Valid loss: {:.3f}".format(lossv))

            #save best model to checkpoint
            if lossv < lossMIN:
                lossMIN = lossv
                model_num = epochID + 1
                torch.save({'epoch': model_num, 'state_dict': model.state_dict(),
                            'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()},
                           f"{output_path}{model_num}-epoch_FL.pth.tar")
                print('Epoch ' + str(model_num) + ' [save] val loss decreased')
                params = model.state_dict() #store parameters of best model
            else:
                print('Epoch ' + str(model_num) + ' [----] val loss did not decrease')

            #store metrics
            save_epoch.append(model_num)
            save_train_loss.append(losst)
            save_val_loss.append(lossv)
            save_val_AUC.append(aurocMean)

        #list of training times
        train_time = np.array(train_end) - np.array(train_start)
        print("Training time for each epoch: {} seconds".format(train_time.round(0)))

        #save logging metrics in CSV
        all_metrics = [save_epoch, train_time, save_train_loss, save_val_loss, save_val_AUC]
        with open(out_csv_path, 'w') as f:
            header = ['epoch', 'time', 'train loss', 'val loss', 'val AUC']
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(zip(*all_metrics))

        #return state dict of best model
        return params

    def epochTrain(model, dataLoaderTrain, optimizer, loss, use_gpu):
        losstrain = 0
        model.train()

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


    def epochVal(model, dataLoaderVal, optimizer, loss, use_gpu):
        model.eval()
        lossVal = 0

        with torch.no_grad():
            for varInput, target in dataLoaderVal:

                if use_gpu:
                    target = target.cuda(non_blocking = True)
                    varInput = varInput.cuda(non_blocking=True)
                    outGT = torch.FloatTensor().cuda()
                    outPRED = torch.FloatTensor().cuda()
                else:
                    outGT = torch.FloatTensor()
                    outPRED = torch.FloatTensor()

                varOutput = model(varInput)

                lossVal += loss(varOutput, target).item()

                #compute AUC mean for validation
                outGT = torch.cat((outGT, target), 0)
                outPRED = torch.cat((outPRED, varOutput), 0)
                aurocIndividual = Trainer.computeAUROC(outGT, outPRED)
                aurocMean = np.array(aurocIndividual).mean()

        return lossVal / len(dataLoaderVal), aurocMean


    def computeAUROC(dataGT, dataPRED):
        # Computes area under ROC curve
        # dataGT: ground truth data
        # dataPRED: predicted data

        outAUROC = []
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        nnClassCount = len(dataGT)

        for i in range(nnClassCount):
            try:
                outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            except ValueError:
                print("AUROC not defined")
                pass
        return outAUROC


    def test(model, dataLoaderTest, class_idx, use_gpu, checkpoint=None):

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


        model.eval()
        with torch.no_grad():
            for i, (varInput, target) in enumerate(dataLoaderTest):

                if use_gpu:
                    target = target.cuda()
                    varInput = varInput.cuda()
                outGT = torch.cat((outGT, target), 0)

                bs, c, h, w = varInput.size() #batchsize, channel, height, width
                varInput = varInput.view(-1, c, h, w) #resize; why?

                out = model(varInput)
                outPRED = torch.cat((outPRED, out), 0)

        aurocIndividual = Trainer.computeAUROC(outGT, outPRED)
        aurocMean = np.array(aurocIndividual).mean()
        print('AUROC mean: {:.4f}'.format(aurocMean))

        for i in range(0, len(aurocIndividual)):
            print(class_names[class_idx[i]], ': {:.4f}'.format(aurocIndividual[i]))

        return outGT, outPRED, aurocMean

class DenseNet121(nn.Module):

    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size, pre_trained=False):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained = pre_trained)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x
