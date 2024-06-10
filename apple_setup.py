import numpy as np
import torch
from transforms import *
from dist_funcs import *
from utils import *

from random import choice

TRAIN_ITERS = 1500
BLUR = False
coarse_steps = 5
depth = 3

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

    rect1 = AxisAlignedRectSDF(p24, p44, p22, p42)

    sdfs=[rect1]

    transforms = [
        SquareToSquareTransformer(p14, p41, p14, p23),
        SquareToSquareTransformer(p14, p41, p24, p33),
        SquareToSquareTransformer(p14, p41, p34, p43),

        SquareToSquareTransformer(p14, p41, p13, p22),
        SquareToSquareTransformer(p14, p41, p23, p32),
        SquareToSquareTransformer(p14, p41, p33, p42),

        SquareToSquareTransformer(p14, p41, p12, p21),
        SquareToSquareTransformer(p14, p41, p22, p31),
        SquareToSquareTransformer(p14, p41, p32, p41)
    ]

    fract = IFStransforms(sdfs, transforms)
    fract.to(dev)
    return pts, fract
"""
    pts = [torch.nn.Parameter(.05 - .1*torch.rand((2,),device=dev)) for i in range(20)]


    sdfs=[
        WeightedDiskSDF(pts, torch.tensor(0.001).to(dev))
    ]



    transforms = [LearnedLineTransformer(pts, device=dev) for i in range(6)]


    fract = IFStransforms(sdfs, transforms)
    fract.to(dev)
    return pts + [param for item in transforms for param in item.parameters()] +[param for item in sdfs for param in item.parameters()], fract






    endpts = [torch.rand((2,))-.5 for i in range(6)]
    endpts = [torch.nn.Parameter(item.float().to(dev)) for item in endpts]

    transforms = [
        LearnedLineTransformer(endpts, device=dev)
        for i in range(len(endpts))
    ]

    #print(transforms[0].get_tmat())
    #quit()
    rads = [ torch.nn.Parameter(torch.tensor(.001).to(dev)) for i in range(len(endpts))]

    sdfs=[WeightedDiskSDF(endpts,rad) for rad, pt in zip(rads,endpts)]

    fract = IFStransforms(sdfs, transforms)
    fract.to(dev)
    return endpts + rads + [param for item in transforms for param in item.parameters()] +[param for item in sdfs for param in item.parameters()], fract

transforms = [
    SquareToSquareTransformer(p14, p41, p14, p23),
    SquareToSquareTransformer(p14, p41, p24, p33),
    SquareToSquareTransformer(p14, p41, p34, p43),

    SquareToSquareTransformer(p14, p41, p13, p22),
    SquareToSquareTransformer(p14, p41, p23, p32),
    SquareToSquareTransformer(p14, p41, p33, p42),

    SquareToSquareTransformer(p14, p41, p12, p21),
    SquareToSquareTransformer(p14, p41, p22, p31),
    SquareToSquareTransformer(p14, p41, p32, p41)
]


"""
