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

from tqdm import tqdm

import sys

NAME = 'koch'
dev = "cuda"

import importlib
#importlib.invalidate_caches()
smod = importlib.import_module(NAME + "_setup")

params, fract = smod.setup(dev, randomize=True)

#fract.load_state_dict(torch.load(NAME + ".pt"))

with torch.no_grad():
    for name, param in fract.named_parameters():
            if 'sdfs.0.e1' in name:
                param.copy_(torch.tensor([-.41, -0.03]).to(dev))
            if 'sdfs.0.e2' in name:
                param.copy_(torch.tensor([-.13, -0.03]).to(dev))
            if 'sdfs.1.e2' in name:
                param.copy_(torch.tensor([0.0, 0.21]).to(dev))
            if 'sdfs.2.e2' in name:
                param.copy_(torch.tensor([.13, -0.03]).to(dev))
            if 'sdfs.3.e2' in name:
                param.copy_(torch.tensor([.41, -0.03]).to(dev))


# These operations will help with our BlurMSE loss function.
pool = torch.nn.MaxPool2d(2,2)
if smod.BLUR:
    blur = GaussianBlur(5, 5.0)
else:
    blur = torch.nn.Identity()

criterion = torch.nn.L1Loss()


K = 16
# Define a target image to train against.
#orig_im = np.sign(imread("koch_V2.png"))#[:,:,0] #np.pad(np.ones((32,32)),((16,16),(16,16)))
img = np.sign(imread(NAME + ".png"))
target = np.sign(img)
#orig_im = np.sign(imread("sierpinski.png"))

# Our initial image size is 256 x 256, so we need coordinates for all the pixels in that square.
coords = get_image_coords(K)
coords_back = coords.copy()
coords = torch.tensor(coords).to(dev)

# Transforming the image to distance representation.
target = 1 - target
target = np.sign(skresize(target, (K,K)))
target = distance_transform_edt(target) # - distance_transform_edt(1-target)
target /= K
target = torch.tensor(target.reshape(1,K,K)).to(dev)
targ = torch.exp( -1.0 * torch.pow(target,2.0) * (K/.025)  )
#print(target.max())
MIN_LOSS = 1e6

losses = np.zeros((coords.shape[0],1+smod.coarse_steps))
vecfield_fine = np.zeros((coords.shape[0],2))
vecfield_coarse = np.zeros((coords.shape[0],2))
vecfield_gt = np.zeros((coords.shape[0],2))

CHOSEN_PARAM_NAME = 'sdfs.1.e2'
CORRECTPT = np.array({name:item for name,item in fract.named_parameters()}[CHOSEN_PARAM_NAME].detach().cpu().numpy())

print(CORRECTPT)
#quit()

for i in tqdm(range(coords.shape[0])):
    FOLLOWPARAM = None
    with torch.no_grad():
        for name, param in fract.named_parameters():
                if CHOSEN_PARAM_NAME in name:
                    param.copy_(torch.flip(coords[i],[0]))
                    FOLLOWPARAM = param

    dists = fract(coords, depth=smod.depth)

    dists = torch.nn.functional.relu(dists)
    transformed_pixels = dists.clone().reshape(1,K,K)
    #plt.matshow(transformed_pixels[0].detach().cpu().numpy())
    #plt.show()
    #quit()
    transformed_pixels = torch.exp( -1.0 * torch.pow(transformed_pixels,2.0) * (K/.025)  )
    #print(dists.max())
    vals = transformed_pixels.clone()
    #vals = (vals / vals.max())
    vals2 = vals.clone()
    loss = 0.0

    targ2 = targ.clone()
    #print(vals2.max(), targ2.max())
    #print(targ.device, vals.device)
    loss += criterion(vals,targ.float())

    losses[i,0] = loss.detach().cpu().numpy()

    vecfield_gt[i, :] = CORRECTPT - coords[i].cpu().numpy()
    vecfield_fine[i,:] = torch.autograd.grad(loss, FOLLOWPARAM, retain_graph=True)[0].cpu().detach().numpy()

    for jj in range(min(smod.coarse_steps,int(np.log2(K))-1)):
        #print(vals2.shape)
        vals2 = pool(torch.clamp(blur(vals2), min=0.0, max= 1000.0))
        targ2 = pool(torch.clamp(blur(targ2), min=0.0, max= 1000.0))
        loss += criterion(vals2,targ2.float())

        losses[i,jj+1] = loss.detach().cpu().numpy() - losses[i,jj]

    vecfield_coarse[i,:] = torch.autograd.grad(loss, FOLLOWPARAM, retain_graph=True)[0].cpu().detach().numpy()

#print(losses)
#print(losses.sum(1).min(0))

#print(vecfield)

vecfield_coarse /= np.linalg.norm(vecfield_coarse,axis=1,keepdims=True)
vecfield_fine /= np.linalg.norm(vecfield_fine,axis=1,keepdims=True)
vecfield_gt /= np.linalg.norm(vecfield_gt,axis=1,keepdims=True)

print("Coarse Alignment:", np.abs(vecfield_coarse * vecfield_gt).sum(1).mean(0))
print("Fine Alignment:", np.abs(vecfield_fine * vecfield_gt).sum(1).mean(0))


X,Y = coords_back.T
Y = Y.reshape(K,K)
X = X.reshape(K,K)

U,V = vecfield_gt.T
U = U.reshape(K,K)
V = V.reshape(K,K)

plt.gca().set_aspect('equal')

plt.imshow(img.T[::-1,:], extent=[-.5, .5, -.5, .5], zorder=1, cmap='Greys',  interpolation='nearest')

plt.quiver(X,Y,U,V, angles='xy', color='orange', label="Ground Truth Direction")

U,V = vecfield_fine.T
U = U.reshape(K,K)
V = V.reshape(K,K)

plt.quiver(X,Y,U,V, angles='xy', color='blue', label="Fine Loss Gradient")

U,V = vecfield_coarse.T
U = U.reshape(K,K)
V = V.reshape(K,K)

plt.quiver(X,Y,U,V, angles='xy', color='green', label="Coarse Loss Gradient")

plt.scatter(CORRECTPT[0], CORRECTPT[1], c='red')

plt.title("Gradient Direction vs Direction to Optimum",fontsize=20)
plt.legend(fontsize=14,framealpha=1.0)
plt.axis('off')
plt.savefig("results/TO_USE/"+NAME+"_vecfield.png",bbox_inches='tight')
plt.show()


quit()

for jj in range(1,losses.shape[1]):
    test = losses[:,:jj].mean(1).copy().reshape(K,K)
    plt.matshow(test)
    #plt.tight_layout()
    plt.axis('off')
    plt.savefig("loss_fig_%d" % jj,bbox_inches='tight')
    plt.show()
