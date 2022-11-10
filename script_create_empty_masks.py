
from __future__ import print_function


import os, json, random, sys, math, torch, copy, hashlib, io
torch.manual_seed(1)
random.seed(1)
import numpy as np
np.random.seed(1)
import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix, multilabel_confusion_matrix
import pandas
import _pickle as cPickle

from datasets import get_dataloader, get_num_classes, get_class_names
from models import get_model

from base_trainer import BaseTrainer
from functools import partial

from opts import get_arguments
from core.config import cfg, cfg_from_file, cfg_from_list
from datasets.utils import Colorize
from losses import get_criterion, mask_loss_ce

from utils.timer import Timer
from utils.stat_manager import StatManager
from torchvision.utils import save_image as sv
from torchvision import transforms

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_color_map
from PIL import Image, ImagePalette

directory = '/mnt/data/chendudai/repos/wikiscenes/test_0_1/images'
path = './data/wikiscenes_test_0_1.txt'

# with open(path, 'w') as f:
#
#     for filename in os.listdir(directory):
#         full_path_images = './test_0_1/images/' + filename + ' '
#         full_path_masks = './test_0_1/masks/' + filename
#
#         f.write(full_path_images)
#         f.write(full_path_masks)
#         f.write('\n')


with open(path) as f:
    lines = f.readlines()

for line in lines:
    path_to_load = line[:26]
    path_to_save = line[27:52]

    if path_to_load[-1] == 'e':
        path_to_load = path_to_load + 'g'


    if path_to_save[-1] == 'p':
        path_to_save = path_to_save[1:] + 'eg'


    with Image.open(path_to_load) as im:
        size_1 = 200
        size_2 = int(np.floor(im.size[1]/(im.size[0]/size_1)))
        im = im.resize((size_1,size_2))
        im.save(path_to_load)


        mask_to_save = np.zeros((im.height, im.width))
        mask_to_save[0][0] = 1
        mask_to_save = Image.fromarray(mask_to_save, mode='L')
        mask_to_save.save(path_to_save)


