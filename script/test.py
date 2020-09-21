import torch
import torch.nn as nn
import torch.optim as optim

import time
import os
import json
import numpy as np
from numpy import tile
import pandas as pd


path = os.listdir('/data4/mjx/GD-B/yolov4/image')
for solo_path in path:
    with open("/data4/mjx/GD-B/yolov4/train.txt", "a")  as f:
        f.writelines('/data4/mjx/GD-B/yolov4/image/'+ solo_path + '\n')
    f.close