import numpy as np
import torch

def adjustLearningRate(optimizer, epoch, args):
    lrAdjust = {epoch: args.learningRate * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lrAdjust.keys():
        lr = lrAdjust[epoch]
        for paramGroup in optimizer.param_groups:
            paramGroup['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.bestScore = None
        self.earlyStop = False
        self.valLossMin = np.inf
        self.delta = delta

    def __call__(self, valLoss, model, path):
        score = -valLoss
        if self.bestScore is None:
            self.bestScore = score
            self.saveCheckpoint(valLoss, model, path)
        elif score < self.bestScore + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.earlyStop = True
        else:
            self.bestScore = score
            self.saveCheckpoint(valLoss, model, path)
            self.counter = 0

    def saveCheckpoint(self, valLoss, model, path):
        print(f'MSE decreased ({self.valLossMin:.6f} --> {valLoss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.valLossMin = valLoss




