import numpy as np

import torch
torch.manual_seed(134994)

import scipy as sp

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

from skimage.transform import resize as skresize
from skimage.io import imread, imsave
from scipy.ndimage import distance_transform_edt

import cv2 as cv

from torchvision.transforms.functional import affine as AffTransform
from torchvision.transforms import GaussianBlur

from utils import *
from constants import *

import sys

# train on the GPU
dev = 'cuda'

NAME = sys.argv[1]

import importlib
#importlib.invalidate_caches()
smod = importlib.import_module(NAME + "_setup")

params, fract = smod.setup(dev)

opt = torch.optim.Adam(params, 0.003)

# These operations will help with our BlurMSE loss function.
pool = torch.nn.MaxPool2d(2,2)
if smod.BLUR:
    blur = GaussianBlur(5, 5.0)
else:
    blur = torch.nn.Identity()

criterion = torch.nn.MSELoss()


K = 256
# Define a target image to train against.
#orig_im = np.sign(imread("koch_V2.png"))#[:,:,0] #np.pad(np.ones((32,32)),((16,16),(16,16)))
target = np.sign(imread(NAME + ".png"))
#orig_im = np.sign(imread("sierpinski.png"))

# Our initial image size is 256 x 256, so we need coordinates for all the pixels in that square.
coords = get_image_coords(K)
coords = torch.tensor(coords).to(dev)

# Transforming the image to distance representation.
target = 1 - target
target = np.sign(skresize(target, (K,K)))
target = distance_transform_edt(target) #- distance_transform_edt(1.0 - target)


target /= K

#print(target.max())
MIN_LOSS = 1e6

for i in range(smod.TRAIN_ITERS):
    opt.zero_grad()

    dists = fract(coords, depth=smod.depth)

    dists = torch.nn.functional.relu(dists)
    transformed_pixels = dists.clone().reshape(1,K,K)
    #print(dists.max())
    vals = transformed_pixels.clone()
    #vals = (vals / vals.max())
    vals2 = vals.clone()
    loss = 0.0

    targ = torch.tensor(target.reshape(1,K,K)).to(dev)
    targ2 = targ.clone()
    #print(vals2.max(), targ2.max())
    #print(targ.device, vals.device)
    loss += criterion(vals,targ.float())

    for jj in range(smod.coarse_steps):#int(np.log2(K))-1):
        #print(vals2.shape)
        vals2 = -1*pool(-1*torch.clamp(blur(vals2), min=0.0, max= 1000.0))
        targ2 = -1*pool(-1*torch.clamp(blur(targ2), min=0.0, max= 1000.0))
        loss += criterion(vals2,targ2.float())
    print(i, loss)#, torch.autograd.grad(loss, params[0], retain_graph=True) )

    ls_numpy = loss.detach().cpu().numpy()
    if ls_numpy < MIN_LOSS:
        torch.save(fract.state_dict(), NAME + ".pt")
        MIN_LOSS = ls_numpy

    loss.backward()
    opt.step()

    transformed_pixels = torch.nn.functional.relu(dists).clone().reshape(1,K,K)

    if False:#i % 25 == 0:
        fig, ax = plt.subplots(1,4)
        # The target image.
        ax[0].matshow(targ.cpu().detach()[0])
        # The produced image.
        ax[1].matshow(vals.cpu().detach()[0])
        # The difference between
        ax[2].matshow(torch.abs(vals-targ).cpu().detach()[0])

        rendered_im = torch.exp( -1.0 * torch.pow(transformed_pixels,2.0) * (K/.025)  )

        ax[3].matshow(rendered_im.cpu().detach()[0])

        [aa.axis('off') for aa in ax]
        plt.show()
    if i % 5 == 0:

        rendered_im = torch.exp( -1.0 * torch.pow(transformed_pixels,2.0) * (K/.0025)  ).cpu().detach()[0].numpy()
        #rendered_im = rendered_im.cpu().detach()[0].numpy()
        rendered_im -= rendered_im.min()
        rendered_im /= rendered_im.max()
        rendered_im = 1.0-rendered_im
        rendered_im = (255*rendered_im).astype(np.uint8)
        #print(rendered_im)
        #print(rendered_im.max(), rendered_im.mean())
        imsave("results/%s/gen%04d_%04d.png" % (NAME,K,i), rendered_im, check_contrast=False)
