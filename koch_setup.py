import numpy as np

import torch
torch.manual_seed(134994)

from transforms import IFStransforms, LineToLineTransformer, IdentityTransformer
from dist_funcs import *
from utils import *

from random import choice

TRAIN_ITERS = 1500
BLUR = False
coarse_steps = 5
depth=3

def setup(dev, randomize=True):

    # .05 - .1*np.random.random(item.shape)
    pts = [item for item in np.linspace([-.4,.05],[.4,.05],5).copy()]
    #pts.insert(2, np.array([0.0,0.05]))
    if randomize:
        pts = [item + .1 - .2*np.random.random(item.shape) for item in pts]


    pts = [torch.tensor(item).float().to(dev) for item in pts]

    pts = [torch.nn.Parameter(item) for item in pts]

    sdfs = [LineSDF(
        pA,
        pB
    ) for pA, pB in zip(pts[:-1], pts[1:])]

    transforms = [
        LineToLineTransformer(pts[0], pts[-1], pts[0], pts[1]),
        LineToLineTransformer(pts[0], pts[-1], pts[1], pts[2]),
        LineToLineTransformer(pts[0], pts[-1], pts[2], pts[3]),
        LineToLineTransformer(pts[0], pts[-1], pts[3], pts[4]),
    ]

    fract = IFStransforms(sdfs, transforms)
    fract.to(dev)
    return pts, fract
