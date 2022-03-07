#Installation on Anaconda
# conda install pytorch
#pip install easydict
#conda install -c pytorch torchvision or conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch for GPU local device
#pip install pandas
#pip install matplotlib
#conda install -c conda-forge opencv
#conda install -c anaconda pillow
#conda activate ailab
#conda install scipy

import urllib.request
from pathlib import Path
import easydict
from easydict import EasyDict
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
from matplotlib import pyplot as plt
import matplotlib
import random
import torch
import sys
import pandas as pd
import misc.transforms as own_transforms
from config import cfg
from models.CC import CrowdCounter
from misc.utils import *

import scipy.io as sio
from PIL import Image, ImageOps


mean_std = ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])
img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])
pil_to_tensor = standard_transforms.ToTensor()

#img_path = './airporttest/SHHBtest.jpg' # the image to be tested with the inference
img_path = './airporttest/airportfromprojectsmallcrop.jpg'
#img_path = './airporttest/TestLivePAI.jpg'
#img_path = './airporttest/airportcongestion1.jpg'

model_path = './trainedmodel/07-CSRNet_all_ep_39_mae_32.6_mse_74.3.pth' # A large training had been done using AlexNet on UCF50 data up to 4000 epochs 07-CSRNet_all_ep_39_mae_32.6_mse_74.3.pth WE_CSRNet_ep_21_mae_16.3_mse_0.0.pth
#model_path = './trainedmodel/crowdcounting_SHHB_MCNN_ep_595_mae_36.6_mse_57.1.pth'

device = torch.device('cpu') #to force to use CPU if local PC execution
model = CrowdCounter(cfg.GPU_ID,'CSRNet') #to be change to the neural network architecture used AlexNet or MCNN aligned with the saved model pth already loaded
#Load the model already trained for the inference. we need to specify that it is on CPU if it is on local PC
model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
model.eval()
#Load the image to test
img = Image.open(img_path)
plt.imshow(img)
#some convertion of the image

if img.mode == 'L':
    img = img.convert('RGB')
img = img_transform(img)
with torch.no_grad():
    img = Variable(img[None,:,:,:])
    pred_map = model.test_forward(img)
# execute the ML on the image and calculate the people based on the density

pred_map = pred_map.cpu().data.numpy()[0,0,:,:]
pred = np.sum(pred_map)/100.0
pred_map = pred_map/np.max(pred_map+1e-20)
print("{} people".format(pred.round()))
