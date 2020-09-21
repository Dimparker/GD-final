import os
import torch
import numpy as np
import torch.utils.data
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2
import math
import glob
import pickle
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
colors=[[np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)]for i in range(100)]
#为了最终实例分割显示明显,定义常见类别为深色
colors[3]=[0,0,255] #car
colors[6]=[0,0,255] #bus
colors[8]=[0,0,255] #truck

class_need = ['car','truck', 'bus']

def compute_distance(center0, center1):
    return math.sqrt(math.pow((center1[0]-center0[0]),2) + math.pow((center1[1]-center0[1]),2))

def demo():
    img_dir='/data4/mjx/gd/datasets'
    test_dir = '/data4/mjx/gd/dataset_raw/test/'
    # load an instance segmentation model pre-trained on COCO
    device = torch.device('cuda:2')
    model = torchvision.models.detection.maskrcnn_resnet50_fpn()
    state_dict = torch.load('maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth')
    model.load_state_dict(state_dict)
    model.to(device)
    # imgs=os.listdir(img_dir)
    imgs = glob.glob(test_dir+'/*/*')
    
    all_info = {}

    for pic_idx in range(len(imgs)):
        
        img_info = {}
        img_info['box_0_min'], img_info['box_1_min'], img_info['box_2_max'], img_info['box_3_max'] = {}, {}, {}, {}
    
        img_info['car_num'] = {}
        img_info['box_center_gap_mean'] = {}
        img_info['box_center_gap_max'] = {}
        img_info['box_center_gap_min'] = {}
        img_info['box_center_gap_std'] = {}
        img_info['box_center_gap_range'] = {}

        img_info['last_distance'] = {}

        img_info['mask_area_mean'] = {}
        img_info['mask_area_max'] = {}
        img_info['mask_area_min'] = {}
        img_info['mask_area_std'] = {}
        img_info['mask_area_range'] = {}
        img_info['mask_area_all'] = {}
        img_info['mask_road']  = {}

        distance = []
        box_0, box_1, box_2, box_3 = [], [], [], []
        box_center = []
        mask_area = []
        car_num_temp = 0
        imgsrc=cv2.imread(imgs[pic_idx])
        all_cls_mask_color = np.zeros_like(imgsrc)
        all_cls_mask_index=np.zeros_like(imgsrc)
        h,w,c = imgsrc.shape
        img = imgsrc / 255.
        img=np.transpose(img, (2, 0, 1))
        img=torch.tensor(img,dtype=torch.float)
        # put the model in evaluation mode
        model.eval()
        
        with torch.no_grad():
            prediction = model([img.to(device)])
            # print(prediction)
            
            scores =prediction[0]['scores']
            for idx,score in enumerate(scores):
                mask=prediction[0]['masks'][idx][0].cpu().numpy()
                mask=mask>0.5
                cls_id=prediction[0]['labels'][idx].item()
                mask_size = (all_cls_mask_index[mask]==1).size
                boxes =prediction[0]['boxes'][idx].cpu().numpy()
                if score > 0.5 and cls_id < 81 \
                                and class_names[cls_id] in class_need \
                                and mask_size < 100000 \
                                and boxes[0] > 0.2 * w:
                    car_num_temp += 1
                    mask_area.append(mask_size)
                    box_0.append(boxes[0])
                    box_1.append(boxes[1])
                    box_2.append(boxes[2])
                    box_3.append(boxes[3])
                    box_center.append(((boxes[0]+boxes[2])/2, (boxes[1]+boxes[3])/2))
                    # labels =prediction[0]['labels']
                    # all_cls_mask_color[mask]=colors[cls_id]
                    # all_cls_mask_index[mask]=1
                    
                    # cv2.rectangle(imgsrc, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (255, 0, 0), 1)
                    # cv2.putText(imgsrc, class_names[cls_id], (int(boxes[0]),int(boxes[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        
        img_name = imgs[pic_idx].split('/')[-1]
        
        img_info['box_0_min'] = min(box_0) if box_0 else 0
        img_info['box_1_min'] = min(box_1) if box_1 else 0
        img_info['box_2_max'] = max(box_2) if box_2 else 0
        img_info['box_3_max'] = max(box_3) if box_3 else 0
        img_info['last_distance'] = h - max(box_3) if box_3 else 0
        img_info['car_num'] =  car_num_temp
        for i in range(len(box_center)):
            for j in range(i+1, len(box_center)):
                distance.append(compute_distance(box_center[i], box_center[j]))
      
        img_info['box_center_gap_min'] = min(distance) if distance else 0
        img_info['box_center_gap_max'] = max(distance) if distance else 0
        img_info['box_center_gap_mean'] = np.mean(distance) if distance else 0
        img_info['box_center_gap_std'] = np.std(distance) if distance else 0
        img_info['box_center_gap_range'] = max(distance) - min(distance) if distance else 0
        

        img_info['mask_area_max'] = max(mask_area) if mask_area else 0
        img_info['mask_area_min'] = min(mask_area) if mask_area else 0
        img_info['mask_area_mean'] = np.mean(mask_area) if mask_area else 0
        img_info['mask_area_std'] = np.std(mask_area) if mask_area else 0
        img_info['mask_area_range'] = max(mask_area) - min(mask_area) if mask_area else 0
        img_info['mask_area_all'] = sum(mask_area) if mask_area else 0
        img_info['mask_road'] = (w*h/2) - sum(mask_area) if mask_area else 0

        all_info[img_name] = img_info
      
    with open('test_dection_info.pkl', 'wb') as f:
        pickle.dump(all_info, f)


        # img_weight=cv2.addWeighted(imgsrc,0.4,all_cls_mask_color,0.6,0)#线性混合
        # all_mask=all_cls_mask_index==1
        # result=np.copy(imgsrc)
        # result[all_mask]=img_weight[all_mask] #只取mask的混合部分
        # union = np.concatenate((imgsrc,result),axis=1)
        # cv2.imwrite(os.path.join('./seg_res',img_name),result)
        # print(box_0_min, box_1_min, box_2_max, box_3_max)
        # break

demo()
