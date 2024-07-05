from typing import List
import random
import numpy as np
import torch
import json
from time import time


# get start time
def get_StartTime():
    return time()


# get time used
def get_timeUsed(start_time:float):
    return time() - start_time


# set random seed for reproducibility
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
"""
原来的有问题
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
"""


# 写json文件(按行存储格式)
def write_json(obj:List, path):
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(json.dumps(obj[0], ensure_ascii=False))
        for i in range(1, len(obj)):
            fp.write("\n")
            fp.write(json.dumps(obj[i], ensure_ascii=False))


# 读json文件(按行存储格式)
def read_json(path):
    obj = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            obj.append(json.loads(line.strip()))
    return obj


# 加载配置文件, yaml格式
def load_config(path):
    pass
# def load_yaml(file):
#     with open(file, "r", encoding="utf-8") as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)
#     return config


# set device
def device_setting(device:str=None):
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device
