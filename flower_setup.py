import numpy as np
import torch
from transforms import *
from dist_funcs import *
from utils import *

from random import choice

TRAIN_ITERS = 1500
BLUR = False
coarse_steps = 5

def setup(dev, randomize=False):
    endpts = [.5 - torch.rand((2,)) for i in range(100)]
    endpts = [torch.nn.Parameter(item.float().to(dev)) for item in endpts]

    transforms = [
        LearnedLineTransformer(endpts, device=dev)
        for i in range(6)
    ]

    #print(transforms[0].get_tmat())
    #quit()

    sdfs=[WeightedAxisAlignedRectSDF(endpts, torch.tensor(.001).to(dev) )]

    fract = IFStransforms(sdfs, transforms)
    fract.to(dev)
    return endpts + [param for item in transforms for param in item.parameters()] +[param for item in sdfs for param in item.parameters()], fract
