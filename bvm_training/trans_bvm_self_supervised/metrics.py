from PIL import Image
import numpy as np
import time

import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from copy import deepcopy

class PRF(nn.Module):
    def __init__(self, Y, steps=255, end=1):
        super(PRF, self).__init__()
        self.thresholds = torch.linspace(0, end, steps=steps)
        # print(self.thresholds)
        self.Y = Y

    def forward(self, _Y):
        TPs = [torch.sum(torch.sum((_Y >= threshold) & (self.Y), -1), -1).float() for threshold in self.thresholds]
        T1s = [torch.sum(torch.sum(_Y >= threshold, -1), -1).float() for threshold in self.thresholds]
        T2 = Tensor.float(torch.sum(torch.sum(self.Y, -1), -1))
        Ps = [(TP / (T1 + 1e-9)).mean() for TP, T1 in zip(TPs, T1s)]
        Rs = [(TP / (T2 + 1e-9)).mean() for TP in TPs]
        Fs = [(1.3 * P * R / (R + 0.3 * P + 1e-9)) for P, R in zip(Ps, Rs)]

        return {'P':Ps, 'R':Rs, 'F':Fs}
        
def getOutPuts(model, DX, args, supervised=False):
    num_img, channel, height, width = DX.shape
    if supervised:
        OutPuts = {'final':np.empty((len(DX), height, width), dtype=np.float32), 'contour':np.empty((len(DX), 5, height, width), dtype=np.float32), 'preds':np.empty((len(DX), 5, height, width), dtype=np.float32), 'time':0.}
    else:
        OutPuts = {'final':np.empty((len(DX), height, width), dtype=np.float32), 'time':0.} 
    t1 = time.time()

    for idx in range(0, len(DX), args.batch):
        ind = min(len(DX), idx + args.batch)
        X = torch.tensor(DX[idx:ind]).cuda(args.ids[0]).float()
        Outs = model(X)

        OutPuts['final'][idx:ind] = torch.sigmoid(Outs['final']).cpu().data.numpy()
        
        if supervised:
            for supervision in ['preds', 'contour']:
                for i, pred in enumerate(Outs[supervision]):
                    pre = F.interpolate(pred.unsqueeze(0), (height, width), mode='bilinear')[0]
                    pre = torch.sigmoid(pre).cpu().data.numpy()
                    OutPuts[supervision][idx:ind, i] = pre
        
        X, Outs, pre = 0, 0, 0

    OutPuts['time'] = (time.time() - t1)

        
    return OutPuts


def mae(preds, labels, th=0.5):
    return np.mean(np.abs(preds - labels))

def maxF(preds, labels):
    preds = torch.tensor(preds)
    labels = torch.tensor(labels, dtype=torch.uint8)
    
    prf = PRF(labels)
    Fs = prf(preds)['F']
    Fs = [F.cpu().data.numpy() for F in Fs]

    prf.to(torch.device('cpu'))
    torch.cuda.empty_cache()
    return max(Fs)

def Normalize(atten):

    a_min, a_max = atten.min(), atten.max()
    atten = (atten - a_min) * 1. / (a_max - a_min) * 255.

    return np.uint8(atten)

##########################################################

def normalize_(array):
    """Normalize the array to the range [0, 1]."""
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val)

def mse(preds, labels, th=0.5):
    preds_normalized = normalize_(preds)
    labels_normalized = normalize_(labels)
    return np.mean((preds_normalized - labels_normalized) ** 2)

def fscore(preds, labels, th=0.5):
    preds = normalize_(preds)
    labels = normalize_(labels)

    labels_bool = labels.astype(bool)
    tmp = preds >= th
    TP = np.sum(tmp & labels_bool)
    T1 = np.sum(tmp)
    T2 = np.sum(labels)
    F = 1.3 * TP / (T1 + 0.3 * T2 + 1e-9)
    return F

def calculate_iou(preds, labels, th=0.5):
    # Ensure the masks are boolean arrays
    preds = normalize_(preds)
    labels = normalize_(labels)
    mask1 = preds >= th
    mask2 = labels.astype(bool)
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    # Calculate IoU
    iou = intersection / union if union != 0 else 0
    return iou