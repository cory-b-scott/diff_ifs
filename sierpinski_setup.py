import numpy as np
import torch
from transforms import *
from dist_funcs import *
from utils import *

TRAIN_ITERS = 1500
BLUR = False
coarse_steps = 5
depth=3

def setup(dev, randomize=False):
    p11 = torch.tensor([-.4,-.4]).float().to(dev)
    p12 = torch.tensor([-.4,-.15]).float().to(dev)
    p13 = torch.tensor([-.4,.15]).float().to(dev)
    p14 = torch.tensor([-.4,.4]).float().to(dev)

    p21 = torch.tensor([-.15,-.4]).float().to(dev)
    p22 = torch.tensor([-.15,-.15]).float().to(dev)
    p23 = torch.tensor([-.15,.15]).float().to(dev)
    p24 = torch.tensor([-.15,.4]).float().to(dev)

    p31 = torch.tensor([.15,-.4]).float().to(dev)
    p32 = torch.tensor([.15,-.15]).float().to(dev)
    p33 = torch.tensor([.15,.15]).float().to(dev)
    p34 = torch.tensor([.15,.4]).float().to(dev)

    p41 = torch.tensor([.4,-.4]).float().to(dev)
    p42 = torch.tensor([.4,-.15]).float().to(dev)
    p43 = torch.tensor([.4,.15]).float().to(dev)
    p44 = torch.tensor([.4,.4]).float().to(dev)

    pts = [p11, p12, p13, p14, p21, p22, p23, p24, p31, p32, p33, p34, p41, p42, p43, p44]
    #pts = [torch.nn.Parameter(item) for item in pts]
    #pts = [p1,p2,p3,p4,p5]
    #pts = [torch.tensor(item).float().to(dev) for item in np.linspace([-.4,0],[.4,0],5).copy()]

    if randomize:
        pts = [item + .1 - .2*torch.rand(item.shape, device=dev) for item in pts]

    pts = [torch.nn.Parameter(item) for item in pts]


    p11, p12, p13, p14, p21, p22, p23, p24, p31, p32, p33, p34, p41, p42, p43, p44 = pts

    rect1 = AxisAlignedRectSDF(p14, p44, p11, p41)

    sdfs=[rect1]

    transforms = [
        SquareToSquareTransformer(p14, p41, p14, p23),
        SquareToSquareTransformer(p14, p41, p24, p33),
        SquareToSquareTransformer(p14, p41, p34, p43),
        SquareToSquareTransformer(p14, p41, p13, p22),
        SquareToSquareTransformer(p14, p41, p33, p42),
        SquareToSquareTransformer(p14, p41, p12, p21),
        SquareToSquareTransformer(p14, p41, p22, p31),
        SquareToSquareTransformer(p14, p41, p32, p41)
    ]


    fract = IFStransforms(sdfs, transforms)
    fract.to(dev)
    return pts, fract
