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

NAME = sys.argv[1]
dev = "cuda"

import importlib
#importlib.invalidate_caches()
smod = importlib.import_module(NAME + "_setup")

params, fract = smod.setup(dev, randomize=True)

batch_size = 2*4092


depth = int(sys.argv[2])

K = int(sys.argv[3])
coords = 5.0*torch.tensor(get_image_coords(K)).to(dev)

fract.eval()

fract.load_state_dict(torch.load(NAME + ".pt"))

ds = torch.utils.data.TensorDataset(coords, torch.ones_like(coords))

loader = torch.utils.data.DataLoader(ds, shuffle=False, drop_last=False, batch_size=batch_size)
values = []
with torch.no_grad():
    for bX, bY in tqdm(loader):
        dists = fract(bX, depth=depth)
        dists = torch.nn.functional.relu(dists)
        values.append( dists )

all_vals = torch.cat(values).reshape((K,K))
print(all_vals.shape)
pixels = torch.exp( -1.0 * torch.pow(all_vals,2.0) * (K/.0025)  ).cpu().detach().numpy()
print(pixels.shape)
plt.matshow(pixels)
plt.show()

plt.matshow(all_vals.detach().cpu().numpy())
plt.show()

distances = all_vals.detach().cpu().numpy()
distances /= distances.max()
distances = (255*distances).astype(np.uint8)

pixels /= pixels.max()
pixels = (255*pixels).astype(np.uint8)

imsave("results/%s_render.png" % NAME, pixels, check_contrast=False )
imsave("results/%s_render_dist.png" % NAME, distances, check_contrast=False )

"""dd = fract.state_dict()
s = np.array(torch.stack([item for item in dd.values()]).cpu().numpy())
plt.scatter(*s.T)
plt.xlim(-.5,.5)
plt.ylim(-.5,.5)
plt.show()
print(s)
print(list(fract.named_parameters()))
quit()"""
