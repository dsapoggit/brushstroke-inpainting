import os
import sys
import torch
import torch as T
import torch.optim as optim
from torchvision.transforms import functional as F
from neural_monitor import monitor as mon
from neural_monitor import logger
import argparse

from param_stroke import BrushStrokeRenderer
import utils
import losses
import get_contours
from torchvision.utils import save_image
from PIL import Image


import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('content_img_file', type=str, help='Content image file')


parser.add_argument('--position_x', default=0, type=int,
                    help='Left uppermost angle if the mask, x coordinate')
parser.add_argument('--position_y', default=0, type=int,
                    help='Left uppermost angle if the mask, y coordinate')
parser.add_argument('--device', '-d', type=str, default='cuda',
                    help='Device to perform stylization. Default: `cuda`.')
                    
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)
    
content_img_file = args.content_img_file
device = torch.device(args.device)

if __name__ == '__main__':
    content_img = utils.image_loader(content_img_file, 256, device)
    
    mask = np.zeros((content_img.size()[2], content_img.size(3)), dtype=bool)
    
    mask[args.position_x:args.position_x + 256, 
         args.position_y : args.position_y + 256] = True
    
    print(mask.shape, content_img.size())
    
    
    im = Image.fromarray(mask)
    im.save(args.content_img_file[:args.content_img_file.rfind('.')] + '_mask.jpg')
    
    
         
         
         
    

    