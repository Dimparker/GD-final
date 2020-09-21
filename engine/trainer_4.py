import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
import os
import json
import numpy as np
from numpy import tile
from sklearn.metrics import f1_score,precision_recall_fscore_support, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.log import get_logger
from utils.compare import compare, count
from utils.lr_scheduler import cos_lr_scheduler, exp_lr_scheduler
from utils.dataset import roadDataset, roadDatasetInfer
from utils.create_dir import create_dir
from cnn_finetune import make_model
from efficientnet_pytorch import EfficientNet
from config.default import cfg
from torchcontrib.optim import SWA


class BASE():

    def __init__(self, cfg):

        self.gpu_id = cfg.SYSTEM.GPU_ID
        self.num_workers = cfg.SYSTEM.NUM_WORKERS
        self.train_dir = cfg.DATASET.TRAIN_DIR
        self.val_dir = cfg.DATASET.VAL_DIR
        self.test_dir = cfg.DATASET.TEST_DIR
        self.sub_dir = cfg.OUTPUT_DIR.SUB_DIR
        self.log_dir = cfg.OUTPUT_DIR.LOG_DIR 
        self.out_dir = cfg.OUTPUT_DIR.OUT_DIR
        self.model_name = cfg.MODEL.MODEL_NAME
        self.train_batch_size = cfg.TRAIN_PARAM.TRAIN_BATCH_SIZE
        self.val_batch_size= cfg.TRAIN_PARAM.VAL_BATCH_SIZE
        self.test_batch_size = cfg.TRAIN_PARAM.TEST_BATCH_SIZE
        self.momentum = cfg.TRAIN_PARAM.MOMENTUM
        self.weight_decay = cfg.TRAIN_PARAM.WEIGHT_DECAY
        self.num_epochs = cfg.TRAIN_PARAM.NUM_EPOCHS
        self.lr = cfg.TRAIN_PARAM.LR
        self.val_interval = cfg.TRAIN_PARAM.VAL_INTERVAl
        self.print_interval = cfg.TRAIN_PARAM.PRINT_INTERVAL
        self.min_save_epoch = cfg.TRAIN_PARAM.MIN_SAVE_EPOCH
        self.real_json = '69.json'
        
        create_dir(self.sub_dir)
        create_dir(self.out_dir)
        create_dir(self.log_dir)
        create_dir(os.path.join(self.sub_dir,self.model_name))
        create_dir(os.path.join(self.out_dir,self.model_name))
        self.logger = get_logger(os.path.join(self.log_dir, self.model_name+'.log'))

    def loaddata(self, train_dir, batch_size, shuffle,is_train=True):

        image_datasets = roadDataset(train_dir,is_train=is_train)
        # dataset_loaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=False, num_workers=self.num_workers,  pin_memory=True)
        dataset_loaders = torch.utils.data.DataLoader(image_datasets, batch_size=4, shuffle=False, num_workers=4)
        data_set_sizes = len(image_datasets)
        return dataset_loaders, data_set_sizes

    def train_model(self, model, criterion, lr_scheduler):

        self.logger.info('Using: {}'.format(self.model_name))
        self.logger.info('Using the GPU: {}'.format(self.gpu_id))
        self.logger.info('start training...')
        train_loss = []
        since = time.time()
        best_acc = 0.0
        model.train(True)
        optimizer = optim.SGD((model.parameters()), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        
        # 余弦退火策略

        if lr_scheduler is cos_lr_scheduler:
            return_lr_scheduler = lr_scheduler(optimizer, 5)
        if lr_scheduler is exp_lr_scheduler:
            return_lr_scheduler = lr_scheduler(optimizer)

        for epoch in range(self.num_epochs):

            begin_time=time.time()
            # data_loaders, dset_sizes = self.loaddata(train_dir=self.train_dir, batch_size=self.train_batch_size, shuffle=False, is_train=True)
            data_loaders, dset_sizes = self.loaddata(train_dir=self.train_dir, batch_size=4, shuffle=False, is_train=True)
            self.logger.info('-' * 10)
            self.logger.info('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            self.logger.info('learning rate:{}'.format(optimizer.param_groups[-1]['lr']))
            self.logger.info('-' * 10)
            running_loss = 0.0
            running_corrects = 0
            count=0
            for i, data in enumerate(data_loaders):
                count+=1
                inputs, labels = data
                labels = labels.type(torch.LongTensor)
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                # print("input_size",inputs.shape)
                _, outputs = model(inputs)
                # print(outputs)
                # print("output_size:", outputs.shape)
                # print("label_size:", labels.shape)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)
                loss.backward()
                optimizer.step()
                                
                if i % self.print_interval == 0 or outputs.size()[0] < self.train_batch_size:
                    spend_time = time.time() - begin_time
                    self.logger.info(' Epoch:{}({}/{}) loss:{:.3f} '.format(epoch, count, dset_sizes // self.train_batch_size,loss.item()))
                    train_loss.append(loss.item())
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if lr_scheduler is exp_lr_scheduler:
                return_lr_scheduler.step()
            if lr_scheduler is cos_lr_scheduler:
                return_lr_scheduler.step(epoch + i / len(data_loaders))  

            val_acc = self.test_model(model, criterion)
            # self.train_infer(model, epoch)
            epoch_loss = running_loss / dset_sizes
            epoch_acc = running_corrects.double() / dset_sizes
                
            # self.logger.info('Epoch:[{}/{}]\t Loss={:.5f}\t Acc={:.3f} epoch_Time:{} min:'.format(epoch , self.num_epochs-1, epoch_loss, epoch_acc, spend_time/60))
            self.logger.info('Epoch:[{}/{}] Loss={:.5f}  Acc={:.3f} Epoch_Time:{} min: ETA: {} hours'.format(epoch , self.num_epochs-1, epoch_loss, epoch_acc, spend_time/60, (self.num_epochs - epoch) * spend_time / 3600))
            if val_acc > best_acc and epoch > self.min_save_epoch:
                best_acc = val_acc
                best_model_wts = model.state_dict()
            if val_acc > 0.999:
                break
            save_dir = os.path.join(self.out_dir,self.model_name)
            model_out_path = save_dir + "/" + '{}_'.format(self.model_name)+str(epoch) + '.pth'
            torch.save(model.module.state_dict(), model_out_path)
        # save best model
        self.logger.info('Best Accuracy: {:.3f}'.format(best_acc))
        model.load_state_dict(best_model_wts)
        model_out_path = save_dir + "/" + '{}_best.pth'.format(self.model_name)
        torch.save(model.module.state_dict(), model_out_path)
        time_elapsed = time.time() - since
        self.logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def test_model(self, model, criterion):

        model.eval()
        running_loss = 0.0
        running_corrects = 0
        cont = 0
        outPre = []
        outLabel = []
        pres_list=[]
        labels_list=[]
        data_loaders, dset_sizes = self.loaddata(train_dir=self.val_dir, batch_size=self.val_batch_size,  shuffle=False, is_train=False)
        for data in data_loaders:
            inputs, labels = data
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.cuda(), labels.cuda()
            _, outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            if cont == 0:
                outPre = outputs.data.cpu()
                outLabel = labels.data.cpu()
            else:
                outPre = torch.cat((outPre, outputs.data.cpu()), 0)
                outLabel = torch.cat((outLabel, labels.data.cpu()), 0)
            pres_list+=preds.cpu().numpy().tolist()
            labels_list+=labels.data.cpu().numpy().tolist()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            cont += 1
        _,_, f_class, _= precision_recall_fscore_support(y_true=labels_list, y_pred=pres_list,labels=[0, 1, 2, 3], average=None)                                                                   
        fper_class = {'畅通': f_class[0], '缓行': f_class[1], '拥堵': f_class[2], "封闭": f_class[3]}
        submit_score = 0.1*f_class[0]+0.2*f_class[1]+0.3*f_class[2] + 0.4*f_class[3]
        self.logger.info('各类单独F1:{}  各类F加权:{}'.format(fper_class, submit_score))
        self.logger.info('val_size: {}  valLoss: {:.4f} valAcc: {:.4f}'.format(dset_sizes, running_loss / dset_sizes, running_corrects.double() / dset_sizes))
        return submit_score


