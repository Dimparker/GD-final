import os
import torch
from torch import nn
from cnn_finetune import make_model
from efficientnet_pytorch import EfficientNet
from torchsummary import summary

class New_Efficient(nn.Module):

    def __init__(self):
        super(New_Efficient, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b5', num_classes=4)
    def forward(self,x):
        B = x.size(0)
        x = self.backbone.extract_features(x)
        x = self.backbone._avg_pooling(x)
        x = x.view(B, -1)
        temp = x
        x = self.backbone._dropout(x)
        x = self.backbone._fc(x)
        return temp, x


