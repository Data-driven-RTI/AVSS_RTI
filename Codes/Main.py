#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from network import MainModel
from postprocess import processData
from evaluation import CalGradDifferenceBatch,CalIoUBatch,CalEDE
from dataloader import RTIDataset
import matplotlib.pyplot as plt
import os


lr = 1e-3
epochs = 80
img_function = nn.MSELoss()
device = "cuda:1"

def visualize(testfile,pretrainmodel,savedir):
    model = torch.load(pretrainmodel,map_location=device)
    testDataset = RTIDataset(testfile)
    testloader = DataLoader(dataset=testDataset,batch_size=256,shuffle=False,num_workers=1,pin_memory=True)
    with torch.no_grad():
        for testidx,(testdata,testground) in enumerate(testloader):
            print("test idx: ",testidx)
            testdata = testdata.to(device)
            testground = testground.to(device)
            test_img = model(testdata)    
            batchSize = testdata.shape[0]
            if testidx == 0:
                for kk in range(batchSize):
                    imgip = test_img[kk].detach().cpu().numpy()
                    imgt = testground[kk].detach().cpu().numpy()
                    plt.imsave(savedir+str(testidx)+str(kk)+".png",imgip)
                    plt.imsave(savedir+str(testidx)+str(kk)+"gt.png",imgt)

def EvaluationFunction(testfile,model):
    model = torch.load(model,map_location=device)
    testDataset = RTIDataset(testfile)
    testloader = DataLoader(dataset=testDataset,batch_size=256,shuffle=False,num_workers=1,pin_memory=True)
    count = 0
    pixel_difference_sum = 0.0
    pixel_ratio_difference_sum = 0.0
    Iou_sum = 0.0
    with torch.no_grad():
        model.eval()
        for testidx,(testdata,testground) in enumerate(testloader):
            testdata = testdata.to(device)
            testground = testground.to(device)
            test_res = model(testdata)
            batchSize = testdata.shape[0]
            test_res_cur = []
            for ii in range(test_res.shape[0]):
                test_res_cur.append(torch.from_numpy(processData(test_res[ii].detach().cpu().numpy())))
            test_res = torch.stack(test_res_cur,dim=0).to(device) 
            count += batchSize
            pixel_difference,pixel_ratio_difference = CalGradDifferenceBatch(test_res,testground)
            pixel_difference_sum += pixel_difference
            pixel_ratio_difference_sum += pixel_ratio_difference
            Iou_sum += CalIoUBatch(test_res,testground)
    pixel_difference_mean = pixel_difference_sum / count
    ede_value = CalEDE(pixel_difference_mean)
    pixel_ratio_difference_mean = pixel_ratio_difference_sum / count
    Iou_mean = Iou_sum / count
    return ede_value,pixel_ratio_difference_mean,Iou_mean.data.item()


def finetune(trainfile,pretrainmodel,strr):
    fine_epochs = 15
    trainset = RTIDataset(trainfile)
    trainloader = DataLoader(trainset,batch_size=256,pin_memory=True,num_workers=4,shuffle=False)
    model = torch.load(pretrainmodel,map_location=device)
    optimzier = optim.Adam(model.parameters(),lr=5e-4,weight_decay=1e-9)
    for name,param in model.named_parameters():
        if "Imaging.fc" in name: 
            print("yes")
            param.requires_grad = True
        else:
            param.requires_grad = False

    for epoch in range(fine_epochs):
        model.train()
        for trainidx,(traindatax,trainground) in enumerate(trainloader):
            traindatax = traindatax.to(device)
            trainground = trainground.to(device)
            Img_x= model(traindatax)
            img_loss = img_function(Img_x,trainground) 
            allLoss = img_loss
            optimzier.zero_grad()
            allLoss.backward()
            optimzier.step()
            print("Epoch: ",epoch, " Train IDX: ",trainidx,
                  " img loss: ",img_loss.data.item()) 
        if epoch == fine_epochs -1:
            torch.save(model,strr+".pth")


def train(trainfile,strr):
    trainset = RTIDataset(trainfile)
    trainloader = DataLoader(trainset,batch_size=64,pin_memory=True,num_workers=4,shuffle=False,drop_last=True)    
    model = MainModel(in_channels=16,feature_length=1024,img_width=360,img_height=360).to(device)
    optimzier = optim.Adam(model.parameters(),lr=lr,weight_decay=1e-9)
    for epoch in range(epochs):
        model.train()
        for trainidx,(traindata,trainground) in enumerate(trainloader):
            traindata = traindata.to(device)
            trainground = trainground.to(device)
            Img_x = model(traindata)
            img_loss = img_function(Img_x,trainground)
            optimzier.zero_grad()
            img_loss.backward()
            optimzier.step()
            print("Epoch: ",epoch, " Train IDX: ",trainidx,
                  " img loss: ",img_loss.data.item())
        
        if epoch == epochs -1:
            torch.save(model,strr+".pth")

if __name__ == "__main__":
    testfile = "../datafiles/Leave3Out/Env3_Test_leave_3_out.txt"
    modelfile = "../Models/Env3_3_out.pth"
    ede_value,pixel_ratio_difference_mean,Iou_mean = EvaluationFunction(testfile,modelfile)
    print("EDE: ",ede_value," RPD: ",pixel_ratio_difference_mean," IoU: ",Iou_mean)
    

    