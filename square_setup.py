import numpy as np
import torch
from transforms import *
from dist_funcs import *
from utils import *

from random import choice

TRAIN_ITERS = 1500
BLUR = False
coarse_steps = 5
depth=3

def setup(dev, randomize=False):

    # .05 - .1*np.random.random(item.shape)
    pts = [item for item in np.linspace([-.03,.03],[.03,-.03],400).copy()]
    #pts = [item + .05 - .1*np.random.random(item.shape) for item in pts]

    pts = [torch.tensor(item).float().to(dev) for item in pts]

    pts = [torch.nn.Parameter(item) for item in pts]

    sdfs = [LineSDF(
        pA,
        pB
    ) for pA, pB in zip(pts[:-1], pts[1:])]

    transforms = [LearnedLineTransformer(pts) for i in range(4)]#LineToLineTransformer(pts[0], pts[-1], pA, pB) for pA,pB in zip(pts[:-1], pts[1:])]
    sdfs = [WeightedDiskSDF(pts, torch.tensor(0.0001).to(dev)) for i in range(1)]

    fract = IFStransforms(sdfs, transforms)
    fract.to(dev)
    return pts, fract
