import torch
from utils import *

class SquareToSquareTransformer(torch.nn.Module):

    def __init__(self, a,b,c,d, device='cpu'):
        super(SquareToSquareTransformer, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def to(self,device):
        super().to(device)
        self.device=device

    def get_tmat(self):
        #print(self.a)
        sq1 = torch.cat([self.a.unsqueeze(0), self.b.unsqueeze(0)])
        sq2 = torch.cat([self.c.unsqueeze(0), self.d.unsqueeze(0)])

        scale1 = torch.linalg.norm(self.a - self.b)
        scale2 = torch.linalg.norm(self.c - self.d)
        scale = scale2 / scale1

        upper1 = sq1.min(0)[0]
        upper2 = sq2.min(0)[0]

        tm = torch.cat(
            [torch.cat([scale*torch.eye(2, device=scale.device), (upper2 - scale*upper1).unsqueeze(1) ],1),
            torch.tensor([[0,0,1]],device=scale.device)
            ],
            0
        )

        return (scale, tm)#get_transformation_params(self.a, self.b, self.c, self.d)

class IdentityTransformer(torch.nn.Module):

    def __init__(self, device='cpu'):
        super(IdentityTransformer, self).__init__()

    def to(self,device):
        super().to(device)
        self.device=device

    def get_tmat(self):
        return (1.0, torch.eye(3, device=self.device))

class LineToLineTransformer(torch.nn.Module):

    def __init__(self, smaj, emaj, smin, emin, device='cpu'):
        super(LineToLineTransformer, self).__init__()
        self.smaj = smaj
        #self.register_parameter(name="smaj", param=self.smaj)
        self.emaj = emaj
        #self.register_parameter(name="emaj", param=self.emaj)
        self.smin = smin
        #self.register_parameter(name="smin", param=self.smin)
        self.emin = emin
        #self.register_parameter(name="emin", param=self.emin)
        self.device=device

    def to(self, device):
        super().to(device)
        self.device=device

    def get_tmat(self):
        return get_transformation_params(self.smaj, self.emaj, self.smin, self.emin)

class LearnedLineTransformer(torch.nn.Module):

    def __init__(self, pts, device='cpu'):
        super(LearnedLineTransformer, self).__init__()
        self.pts = torch.nn.ParameterList(pts)
        self.assignment = torch.nn.Parameter(torch.rand(4,len(self.pts)).float().to(device))

    def to(self, device):
        super().to(device)
        self.device=device

    def get_tmat(self):
        #print(self.pts[0].device, self.assignment.device)
        chosen = torch.matmul(torch.softmax(self.assignment,1), torch.stack([item for item in self.pts]))
        #print(chosen)
        return get_transformation_params(chosen[0], chosen[1], chosen[2], chosen[3])


class AffineTransformer(torch.nn.Module):

    def __init__(self, device='cpu'):
        super(AffineTransformer, self).__init__()
        self.theta = torch.nn.Parameter(torch.rand(1)*2*tpi)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.tx = torch.nn.Parameter(.5 - torch.rand(1))
        self.ty = torch.nn.Parameter(.5 - torch.rand(1))
        self.device=device

    def to(self, device):
        super().to(device)
        self.device=device

    def get_tmat(self):
        #print(self.device, self.scale.device, self.theta.device,torch.cos(self.theta).device)
        Rm = torch.stack([
             torch.cat([self.scale*torch.cos(self.theta), -1*self.scale*torch.sin(self.theta), self.tx]),
             torch.cat([self.scale*torch.sin(self.theta), self.scale*torch.cos(self.theta), self.ty]),
             torch.tensor([0, 0, 1],device=self.scale.device)
             ]
        )
        #print(self.scale, self.theta)
        #print(Rm)
        return Rm

    def assign_from_mat(self, mat):
        self.tx = torch.nn.Parameter(mat[0,2].reshape((1,)))
        self.ty = torch.nn.Parameter(mat[1,2].reshape((1,)))
        self.scale = torch.nn.Parameter(torch.sqrt(mat[0,0]**2.0 + mat[0,1]**2.0).reshape((1,)))
        self.theta = torch.nn.Parameter(torch.acos(mat[0,0]/self.scale).reshape((1,)))
        #print(self.scale)

class IFStransforms(torch.nn.Module):

    def __init__(self, sdfs, affines, device='cpu'):
        super(IFStransforms, self).__init__()
        self.sdfs = torch.nn.ModuleList(sdfs)
        self.affines = torch.nn.ModuleList(affines)
        self.device= device

    def to(self,device):
        super().to(device)
        for item in self.sdfs:
            item.to(device)
        for item in self.affines:
            item.to(device)
        self.device=device

    def forward(self, query, depth=3):
        self.aft_mats = [item.get_tmat() for item in self.affines]
        self.back_aft_mats = [(1/item[0],torch.linalg.inv(item[1])) for item in self.aft_mats]
        self.transforms = [(1.0,torch.eye(3,device=self.device))]
        self.back_transforms = [(1.0,torch.eye(3,device=self.device))]

        transform_reps = depth

        while transform_reps > 0:
            self.transforms = [(s1*s2,torch.matmul(Tm1, Tm2)) for s1,Tm1 in self.transforms for s2,Tm2 in self.aft_mats]
            self.back_transforms = [(s1*s2,torch.matmul(Tm1, Tm2)) for s1,Tm1 in self.back_transforms for s2,Tm2 in self.back_aft_mats]
            transform_reps -= 1


        query_padded = torch.cat([query, torch.ones(query.shape[0], 1, device=query.device)], axis=1).float()
        query_transformed = [torch.matmul(Tm.float(), query_padded.T).T for _,Tm in self.back_transforms]
        scales_as_query = [s for s,_ in self.back_transforms]

        #print(len(scales_as_query))
        #print(len(query_transformed))
        #quit()
        #testttt = torch.cat(query_transformed).detach().cpu().numpy()
        #print(testttt.shape)
        #plt.scatter(*testttt[::5000,:2].T)
        #plt.show()
        dists = torch.stack([
            torch.stack([
                (1.0/scale)*sdf(query_part[:,:2])
                for sdf in self.sdfs
            ])
            for scale,query_part in zip(scales_as_query,query_transformed)
        ])
        #dists = torch.nn.functional.relu(dists)
        #print(dists.shape)
        dists = dists.min(0)[0]
        #print(dists.shape)
        dists = dists.min(0)[0]
        #print(dists.shape)
        #print(dists)
        #quit()
        return dists
