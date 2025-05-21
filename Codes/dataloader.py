#encoding=utf-8

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np

def dataprocess15(matrix):
    N, M, M = matrix.shape
    diag_mask = torch.eye(M, dtype=torch.bool).unsqueeze(0).expand(N, -1, -1)
    new_matrix = matrix[~diag_mask].view(N, M, -1)
    return new_matrix

class RTIDataset(Dataset):
    def __init__(self,filename):
        super().__init__()
        self.filenamelist = []
        self.groundlist = []
        with open(filename,"r",encoding="utf-8") as f:
            flines = f.readlines()
            for line in flines:
                line = line.strip().split("\t")
                self.filenamelist.append(line[0])
                self.groundlist.append(line[1])

    def __getitem__(self, index):
        data = torch.load(self.filenamelist[index]).float()
        data = dataprocess15(data)
        ground = torch.from_numpy(np.load(self.groundlist[index])).float()
        return data,ground
    
    def __len__(self):
        return len(self.filenamelist)

        