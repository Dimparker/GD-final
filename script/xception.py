import os
import torch
from torch import nn
from utils.compare import compare, count
from utils.lr_scheduler import cos_lr_scheduler, exp_lr_scheduler
from cnn_finetune import make_model
from efficientnet_pytorch import EfficientNet
from config.default import cfg
from engine import trainer, trainer_merge, get_feature_trainer
from models.new_xception import New_Xception
from models.new_efficient import New_Efficient

cfg.merge_from_file('config/efficient.yaml')
torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(','.join(str(i) for i in cfg.SYSTEM.GPU_ID))
model = New_Xception()
# model = make_model('{}'.format(cfg.MODEL.MODEL_NAME), num_classes=3, pretrained=True)
weights_c = torch.FloatTensor([0.2,0.2,0.6])
criterion = nn.CrossEntropyLoss(weight=weights_c).cuda()
# criterion = CrossEntropyLabelSmooth().cuda()
model = nn.DataParallel(model).cuda()
trainer_engine = get_feature_trainer.BASE(cfg)
trainer_engine.train_model(model, criterion, lr_scheduler=cos_lr_scheduler)
torch.cuda.empty_cache()