import os
import argparse as arg

from config.default import cfg
cfg.merge_from_file('config/config.yaml')
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(','.join(str(i) for i in cfg.SYSTEM.GPU_ID))
import torch
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from utils.compare import compare, count
from utils.lr_scheduler import cos_lr_scheduler, exp_lr_scheduler
from utils.seed import set_seed
from cnn_finetune import make_model
from efficientnet_pytorch import EfficientNet
from engine import trainer,trainer_solo, trainer_merge,trainer_dd, get_feature_trainer, get_feature_trainer_merge
from models.new_xception import New_Xception
from models.new_efficient import New_Efficient

set_seed(2020)
parser = arg.ArgumentParser()
parser.add_argument("--local_rank", type=int, default = -1)
args = parser.parse_args()

torch.cuda.empty_cache()
torch.cuda.set_device(args.local_rank)
device = torch.device('cuda', args.local_rank)
torch.distributed.init_process_group(backend='nccl')



if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()  
model = make_model('xception', num_classes=3, pretrained=True)
if args.local_rank == 0:
    torch.distributed.barrier()  
model = model.to(device)
model = torch.nn.parallel.DistributedDataParallel(model, 
                                            device_ids=[args.local_rank], 
                                            output_device=args.local_rank, 
                                            find_unused_parameters=True
                                            )
weights_c = torch.FloatTensor([1,1,3])
criterion = nn.CrossEntropyLoss(weight=weights_c).cuda()
# criterion = CrossEntropyLabelSmooth().cuda()
trainer_engine = trainer_dd.BASE(cfg)
trainer_engine.train_model(model, criterion, lr_scheduler=cos_lr_scheduler)
torch.cuda.empty_cache()