import cv2
import os
import glob
img_paths = sorted(glob.glob('/data4/mjx/gd/all_raw/val/*/*'))
img_writeDir='/data4/mjx/gd/datasetCrop/val'
i=0
for img_path in img_paths:
    i+=1
    print(img_path,i)
    name=img_path.split('/')[-1]
    seq=img_path.split('/')[-2]
   
    target_dir=os.path.join(img_writeDir,seq)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    img = cv2.imread(img_path)
    h,w,_=img.shape
    cropped = img[int(0.3*h):h,int(0.25*w):int(0.75*w)]  # 裁剪坐标为[y0:y1, x0:x1]
    cv2.imwrite(os.path.join(target_dir,name), cropped)