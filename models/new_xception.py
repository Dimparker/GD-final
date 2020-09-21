import os
import torch
from torch import nn
from cnn_finetune import make_model
from efficientnet_pytorch import EfficientNet

class New_Xception(nn.Module):
    
    def __init__(self):
        super(New_Xception, self).__init__()
        self.backbone = make_model('xception', num_classes=3, pretrained=True)
    def forward(self,x):
        x = self.backbone._features(x)
        x = self.backbone.pool(x)
        B,C,H,W = x.shape
        x = x.reshape(B, -1)
        temp = x
        output = self.backbone._classifier(x)
        return temp, output

