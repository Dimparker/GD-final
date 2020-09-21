from yacs.config import CfgNode as CN

_C = CN()
# ---------------------------------------------
# 配置系统信息
# ---------------------------------------------
_C.SYSTEM = CN()  # 创建系统节点
_C.SYSTEM.GPU_ID = 8,9
_C.SYSTEM.NUM_WORKERS = 4  
# ---------------------------------------------
# 配置模型信息
# ---------------------------------------------
_C.MODEL = CN()
_C.MODEL.MODEL_NAME = 'xception'
# ---------------------------------------------
# 配置训练文件夹
# ---------------------------------------------
_C.DATASET = CN()
_C.DATASET.TRAIN_DIR = 'dataset_raw/train'
_C.DATASET.VAL_DIR = 'dataset_raw/val'
_C.DATASET.TEST_DIR = 'dataset_raw/test'
# ---------------------------------------------
# 配置输出文件夹
# ---------------------------------------------
_C.OUTPUT_DIR = CN()
_C.OUTPUT_DIR.SUB_DIR= 'submit/'
_C.OUTPUT_DIR.LOG_DIR= 'log/'
_C.OUTPUT_DIR.OUT_DIR= 'output/'
# ---------------------------------------------
# 配置训练超参
# ---------------------------------------------
_C.TRAIN_PARAM = CN() 
_C.TRAIN_PARAM.IMAGE_SIZE= 214
_C.TRAIN_PARAM.TRAIN_BATCH_SIZE = 8
_C.TRAIN_PARAM.VAL_BATCH_SIZE= 4
_C.TRAIN_PARAM.TEST_BATCH_SIZE = 4
_C.TRAIN_PARAM.MOMENTUM = 0.9
_C.TRAIN_PARAM.WEIGHT_DECAY = 0.0004
_C.TRAIN_PARAM.NUM_EPOCHS = 30
_C.TRAIN_PARAM.LR = 0.002
_C.TRAIN_PARAM.VAL_INTERVAl = 0
# 打印间隔STEP
_C.TRAIN_PARAM.PRINT_INTERVAL = 100
# 最低保存模型/计算最优模型epohc阈值
_C.TRAIN_PARAM.MIN_SAVE_EPOCH= 1

def get_cfg_defaults():
  return _C.clone()
  
cfg = _C 

 