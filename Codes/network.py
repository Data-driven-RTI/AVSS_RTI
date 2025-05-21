#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderModule(nn.Module):
    def __init__(self,in_channels):
        super(EncoderModule,self).__init__()
        self.in_channels = in_channels
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=128,kernel_size=(3,3),stride=1),
            nn.BatchNorm2d(128,track_running_stats=False,affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(5,5),stride=1),
            nn.BatchNorm2d(256,track_running_stats=False,affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(5,5),stride=1),
            nn.BatchNorm2d(512,track_running_stats=False,affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=(6,5),stride=1),
        )

        self.mid_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=256,kernel_size=(9,9),stride=1),
            nn.BatchNorm2d(256,track_running_stats=False,affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(7,7),stride=1),
            nn.BatchNorm2d(512,track_running_stats=False,affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=(2,1),stride=1),
        )

        self.global_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=512,kernel_size=(16,15),stride=1),
            nn.BatchNorm2d(512,track_running_stats=False,affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=(1,1),stride=1),
        )
    
    def forward(self,data):
        global_feature_map = self.global_conv(data)
        local_feature_map = self.local_conv(data)
        middle_feature_map = self.mid_conv(data)
        res_feature = global_feature_map + local_feature_map + middle_feature_map
        res_feature = torch.flatten(res_feature,start_dim=1)
        return res_feature
    
class ImagingModule(nn.Module):
    def __init__(self,feature_length,img_width,img_height):
        super(ImagingModule,self).__init__()
        self.feature_length = feature_length
        self.img_width = img_width
        self.img_height = img_height
        self.fc = nn.Linear(feature_length,1600)
        self.attention = nn.Parameter(torch.ones(size=[40,40]),requires_grad=True)
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(3,3),stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(1,1),stride=1)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=1,kernel_size=(1,1),stride=1)
        self.relu = nn.LeakyReLU()
    def forward(self,X):
        res = self.relu(self.fc(X))
        res = res.reshape(-1,1,40,40)
        res = res * self.attention
        res = F.interpolate(res,scale_factor=3,mode="bilinear")
        res = self.relu(self.conv1(res)) # 120,120
        res = F.interpolate(res,scale_factor=3,mode="bilinear")
        res = self.relu(self.conv2(res)) # 360,360
        res = self.conv3(res)
        return res.squeeze(1)
    

class MainModel(nn.Module):
    def __init__(self,in_channels,feature_length,img_width,img_height) -> None:
        super(MainModel,self).__init__()
        self.Encoder = EncoderModule(in_channels=in_channels)
        self.Imaging = ImagingModule(feature_length,img_width,img_height)
    def forward(self,X):
        FX = self.Encoder(X)
        Img_x = self.Imaging(FX)
        return Img_x 




        
    


