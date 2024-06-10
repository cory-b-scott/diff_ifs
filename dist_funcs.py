import torch
from utils import *

class DiskSDF(torch.nn.Module):

    def __init__(self, center, rad, device='cpu'):
        super(DiskSDF, self).__init__()
        self.rad = rad
        self.center = center
        self.device = device
        try:
            self.register_parameter(name="center", param=self.center)
        except:
            pass
        try:
            self.register_parameter(name="rad", param=self.rad)
        except:
            pass

    def forward(self,query):
        query_cent = query - self.center
        d_from_cent = torch.linalg.norm(query_cent,axis=1)
        return torch.nn.functional.relu(d_from_cent - self.rad)

class WeightedDiskSDF(torch.nn.Module):

    def __init__(self, centers, rad, device='cpu'):
        super(WeightedDiskSDF, self).__init__()
        self.rad = rad
        self.pts = torch.nn.ParameterList(centers)
        self.device = device
        self.assignment = torch.nn.Parameter(torch.rand(1,len(self.pts)).float().to(device))

    def forward(self,query):
        center = torch.matmul(torch.softmax(self.assignment,1), torch.stack([item for item in self.pts]))
        #print(center.shape)
        query_cent = query - center
        d_from_cent = torch.linalg.norm(query_cent,axis=1)
        return torch.nn.functional.relu(d_from_cent - self.rad)


class LineSDF(torch.nn.Module):

    def __init__(self, ep1, ep2, device='cpu'):
        super(LineSDF, self).__init__()
        self.e1 = ep1
        self.e2 = ep2
        self.device = device
        try:
            self.register_parameter(name="e1", param=self.e1)
        except:
            pass
        try:
            self.register_parameter(name="e2", param=self.e2)
        except:
            pass

    def forward(self, query):
        # Given an array of query points and a line segment (p1, p2),
        # calculate the min distance fro meach query point to the line.

        x1, y1 = self.e1
        x2, y2 = self.e2
        px = x2 - x1
        py = y2 - y1

        norm = px*px + py*py
        #print(norm)
        # u is the porportion along the line segment where the
        # perpendicular to query hits; if the perpendicular doesn't
        # hit the line segment this expression reduces to 0 or 1 and picks
        # one of the endpoints.
        u = ((query[:,0] - x1) * px + (query[:,1] -  y1) * py) / norm
        #u = torch.matmul(query-self.e1, (self.e2 - self.e1).T)/norm
        u =  torch.clamp(u, min=0.0, max=1.0)
        #print(u.shape)

        x =  x1 + u * px
        y =  y1 + u * py

        dxy = torch.stack([x,y], axis=1) - query
        dists1 = torch.linalg.norm(dxy, axis=1)
        """
        pa = query - self.e1

        ba = self.e2 - self.e1
        #print(torch.matmul(pa, ba)/torch.dot(ba, ba))
        h = torch.clamp(torch.matmul(pa, ba)/torch.dot(ba, ba), min=0.0, max=1.0)
        #print(h)
        #print(u)
        dists2 = torch.linalg.norm(pa - torch.matmul(h.unsqueeze(1),ba.unsqueeze(0)),axis=1)
        #print(dists1)
        """
        return dists1 - 0.005

class WeightedLineSDF(torch.nn.Module):

    def __init__(self, centers, device='cpu'):
        super(WeightedLineSDF, self).__init__()
        self.pts = torch.nn.ParameterList(centers)
        self.device = device
        self.assignment = torch.nn.Parameter(torch.rand(2,len(self.pts)).float().to(device))

    def forward(self, query):
        e1, e2 = torch.matmul(torch.softmax(self.assignment,1), torch.stack([item for item in self.pts]))
        #print(center.shape)

        # Given an array of query points and a line segment (p1, p2),
        # calculate the min distance fro meach query point to the line.

        x1, y1 = e1
        x2, y2 = e2
        px = x2 - x1
        py = y2 - y1

        norm = px*px + py*py
        #print(norm)
        # u is the porportion along the line segment where the
        # perpendicular to query hits; if the perpendicular doesn't
        # hit the line segment this expression reduces to 0 or 1 and picks
        # one of the endpoints.
        u = ((query[:,0] - x1) * px + (query[:,1] -  y1) * py) / norm
        #u = torch.matmul(query-self.e1, (self.e2 - self.e1).T)/norm
        u =  torch.clamp(u, min=0.0, max=1.0)
        #print(u.shape)

        x =  x1 + u * px
        y =  y1 + u * py

        dxy = torch.stack([x,y], axis=1) - query
        dists1 = torch.linalg.norm(dxy, axis=1)
        """
        pa = query - self.e1

        ba = self.e2 - self.e1
        #print(torch.matmul(pa, ba)/torch.dot(ba, ba))
        h = torch.clamp(torch.matmul(pa, ba)/torch.dot(ba, ba), min=0.0, max=1.0)
        #print(h)
        #print(u)
        dists2 = torch.linalg.norm(pa - torch.matmul(h.unsqueeze(1),ba.unsqueeze(0)),axis=1)
        #print(dists1)
        """
        return dists1

#class RectSDF(torch.nn.Module):
#

class AxisAlignedRectSDF(torch.nn.Module):


    def __init__(self, c11, c12, c21, c22, device='cpu'):
        super(AxisAlignedRectSDF, self).__init__()
        self.c11 = c11
        self.c12 = c12
        self.c21 = c21
        self.c22 = c22

    def forward(self, query):
        center = .5*(self.c22 + self.c11)
        #print(center)
        R = torch.abs(self.c22 - center)
        #print(R)
        p = query - center
        q = torch.abs(p)-R
        dists = torch.linalg.norm(torch.nn.functional.relu(q),axis=1) - torch.nn.functional.relu(-q.max(1)[0])
        #print(dists)
        return dists
        #print(query)


class WeightedAxisAlignedRectSDF(torch.nn.Module):


    def __init__(self, centers, device='cpu'):
        super(WeightedAxisAlignedRectSDF, self).__init__()
        self.pts = torch.nn.ParameterList(centers)
        self.device = device
        self.assignment = torch.nn.Parameter(torch.rand(2,len(self.pts)).float().to(device))

    def forward(self, query):
        c11, c22 = torch.matmul(torch.softmax(self.assignment,1), torch.stack([item for item in self.pts]))
        center = .5*(c22 + c11)
        #print(center)
        R = torch.abs(c22 - center)
        #print(R)
        p = query - center
        q = torch.abs(p)-R
        dists = torch.linalg.norm(torch.nn.functional.relu(q),axis=1) - torch.nn.functional.relu(-q.max(1)[0])
        #print(dists)
        return dists
        #print(query)

class NeuralSDF(torch.nn.Module):

    def __init__(self, sdf1, device='cpu'):
        super(NeuralSDF, self).__init__()
        self.subsdf = sdf1
        self.model = torch.nn.Sequential(
            torch.nn.Linear(3,2),
            torch.nn.ELU(),
            torch.nn.Linear(2,1)
        )

    def forward(self,query):
        sub_dists = self.subsdf(query)
        model_dists = self.model(torch.cat([query, sub_dists.unsqueeze(-1)], dim=-1))
        return torch.nn.functional.relu(sub_dists) + torch.mul(model_dists, -1* torch.nn.functional.relu(-1*sub_dists))


class DifferenceSDF(torch.nn.Module):

    def __init__(self, sdf1, sdf2, device='cpu'):
        super(DifferenceSDF, self).__init__()
        self.sdf1 = sdf1
        self.sdf2 = sdf2

    def forward(self, query):
        stacked = torch.stack([self.sdf1(query), -1.0*self.sdf2(query)],dim=-1)
        #print(stacked.shape)
        return stacked.max(-1)[0]

class UnionSDF(torch.nn.Module):

    def __init__(self, sdf1, sdf2, device='cpu'):
        super(UnionSDF, self).__init__()
        self.sdf1 = sdf1
        self.sdf2 = sdf2

    def forward(self, query):
        stacked = torch.stack([self.sdf1(query), self.sdf2(query)],dim=-1)
        #print(stacked.shape)
        return stacked.min(-1)[0]

class ListUnionSDF(torch.nn.Module):

    def __init__(self, multiple_sdfs, device='cpu'):
        super(ListUnionSDF, self).__init__()
        self.sdfs = torch.nn.ModuleList([item for item in multiple_sdfs])

    def forward(self, query):
        result = self.sdfs[0](query)
        for sdf in self.sdfs[1:]:
            result = torch.minimum(result, sdf(query))
        #print(stacked.shape)
        return result

class IntersectionSDF(torch.nn.Module):

    def __init__(self, sdf1, sdf2, device='cpu'):
        super(IntersectionSDF, self).__init__()
        self.sdf1 = sdf1
        self.sdf2 = sdf2

    def forward(self, query):
        stacked = torch.stack([self.sdf1(query), self.sdf2(query)],dim=-1)
        #print(stacked.shape)
        return stacked.max(-1)[0]

class ListIntersectionSDF(torch.nn.Module):

    def __init__(self, multiple_sdfs, device='cpu'):
        super(ListIntersectionSDF, self).__init__()
        self.sdfs = torch.nn.ModuleList([item for item in multiple_sdfs])

    def forward(self, query):
        result = self.sdfs[0](query)
        for sdf in self.sdfs[1:]:
            result = torch.maximum(result, sdf(query))
        #print(stacked.shape)
        return result

class XorSDF(torch.nn.Module):

    def __init__(self, sdf1, sdf2, device='cpu'):
        super(XorSDF, self).__init__()
        self.sdf1 = sdf1
        self.sdf2 = sdf2

    def forward(self, query):
        stacked = torch.stack([self.sdf1(query), self.sdf2(query)],dim=-1)
        doublestacked = torch.stack( [-1*stacked.max(-1)[0], stacked.min(-1)[0]] , dim= -1)
        return doublestacked.max(-1)[0]
