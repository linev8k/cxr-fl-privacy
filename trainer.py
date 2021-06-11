"""Module for training"""

import time
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


import torch
import torchvision
from torch import nn
from torch import optim
from torch.backends import cudnn

#TODO
#progress bar
#moving to cuda needed in epoch and test functions?
#monitor training progress: loss, val accuracy (?)


class Trainer():

    def train(model, dataLoaderTrain, dataLoaderVal, nnClassCount, cfg, output_path, use_gpu, checkpoint=None):


        """Train a model.
        Args
            model: Instance of the model to be trained.
            dataLoaderTrain: Dataloader for training data.
            dataLoaderVal: Dataloader for validation data.
            nnClassCount: Number of labels to be trained on.
            cfg: Config dictionary containing training parameters.
            output_path: Path where results should be saved.
            use_gpu: Whether to use available GPUs.
            checkpoint: A model checkpoint to load from for continuing training.
        """

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

        for epochID in range(0, cfg['max_epochs']):
            train_start.append(time.time()) # training starts
            losst = Trainer.epochTrain(model, dataLoaderTrain, optimizer, loss, use_gpu)
            train_end.append(time.time()) # training ends

            #validation
            lossv = Trainer.epochVal(model, dataLoaderVal, optimizer, loss, use_gpu)
            print("Training loss: {:.3f},".format(losst), "Valid loss: {:.3f}".format(lossv))

            #save best model
            if lossv < lossMIN:
                lossMIN = lossv
                model_num = epochID + 1
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(),
                            'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()},
                           f"{output_path}{epochID + 1}-epoch_FL.pth.tar")
                print('Epoch ' + str(epochID + 1) + ' [save] val loss decreased')
            else:
                print('Epoch ' + str(epochID + 1) + ' [----] val loss did not decrease')

            print('\n')
            params = model.state_dict()

        #list of training times
        train_time = np.array(train_end) - np.array(train_start)
        print("Training time for each epoch: {} seconds".format(train_time.round(0)))
        print('\n')

        #epoch with lowest validation loss, state dict of best model
        return model_num, params

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

                varOutput = model(varInput)

                lossVal += loss(varOutput, target).item()

        return lossVal / len(dataLoaderVal)


    def computeAUROC(dataGT, dataPRED, nnclassCount):
        # Computes area under ROC curve
        # dataGT: ground truth data
        # dataPRED: predicted data

        outAUROC = []
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()

        for i in range(nnclassCount):
            try:
                outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            except ValueError:
                print("AUROC not defined")
                pass
        return outAUROC


    def test(model, dataLoaderTest, nnClassCount, class_names, use_gpu, checkpoint=None):

        if use_gpu:
            cudnn.benchmark = True #select fastest conv. algorithm
            outGT = torch.FloatTensor().cuda()
            outPRED = torch.FloatTensor().cuda()
        else:
            outGT = torch.FloatTensor()
            outPRED = torch.FloatTensor()

        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])


        model.eval()
        outPROB = []
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

        aurocIndividual = Trainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        print('AUROC mean: {:.4f}'.format(aurocMean))

        for i in range (0, len(aurocIndividual)):
            print(class_names[i], ': {:.4f}'.format(aurocIndividual[i]))

        return outGT, outPRED

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
