import os
from config.default import cfg

import torch
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from utils.compare import compare, count
from utils.lr_scheduler import cos_lr_scheduler, exp_lr_scheduler
from utils.seed import set_seed
from cnn_finetune import make_model
from efficientnet_pytorch import EfficientNet
from engine import trainer_4
from utils.CELS import CrossEntropyLabelSmooth

class New_Efficient2(nn.Module):

    def __init__(self):
        super(New_Efficient2, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b5', num_classes=4)
    def forward(self,x):
      
        B = x.size(0)
        x = self.backbone.extract_features(x)
        x = self.backbone._avg_pooling(x)
        x = x.view(B, -1)
        temp = x
        x = torch.sum(x, dim=0, keepdim=True)
        # x = x.expand(4, -1)
        x = self.backbone._dropout(x)
        x = self.backbone._fc(x)
        return temp, x

# model = New_Efficient2()
# x = torch.rand((4,3,500,900))
# print(x.shape)
# h,j = model(x)
# print(j.shape)

cfg.merge_from_file('config/efficient_4.yaml')
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(','.join(str(i) for i in cfg.SYSTEM.GPU_ID))
torch.cuda.empty_cache()
model = New_Efficient2()
weights_c = torch.FloatTensor([0.1,0.2,0.3,0.4])
criterion = nn.CrossEntropyLoss(weight=weights_c).cuda()
# criterion = CrossEntropyLabelSmooth().cuda()
model = nn.DataParallel(model).cuda()
trainer_engine = trainer_4.BASE(cfg)
trainer_engine.train_model(model, criterion, lr_scheduler=cos_lr_scheduler)
torch.cuda.empty_cache()