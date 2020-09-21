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
from engine import trainer, trainer_swa
from models.new_efficient import New_Efficient

cfg.merge_from_file('config/efficient_no.yaml')
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(','.join(str(i) for i in cfg.SYSTEM.GPU_ID))
torch.cuda.empty_cache()
# model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=4)
model = New_Efficient()
weights_c = torch.FloatTensor([0.1,0.2,0.3,0.4])
criterion = nn.CrossEntropyLoss(weight=weights_c).cuda()
# criterion = CrossEntropyLabelSmooth().cuda()
model = nn.DataParallel(model).cuda()
trainer_engine = trainer.BASE(cfg)
trainer_engine.train_model(model, criterion, lr_scheduler=cos_lr_scheduler)
torch.cuda.empty_cache()