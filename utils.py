import torch
import numpy as np

def get_image_coords(K):
    t = (1/(K-1))*(np.mgrid[:K, :K])
    coords = np.stack((t[0], t[1]), axis=2).reshape(K*K,2)
    coords -= coords.mean(0)
    return coords

def get_transformation_params(end1in, end2in, p1, p2, direction=torch.tensor(1.0)):
    #print([(item,item.device) for item in self.parameters()])
    #print(end1in.device, end2in.device, p1.device, p2.device)
    # Using sigmoid(direction) as a differentiable stand-in for sign(direction).
    dir_mod = torch.sigmoid(1e2*direction)
    # Swap the two endpoints depending on the sign of direction.
    end1 = dir_mod * end1in + ((1-dir_mod) * end2in)
    end2 = dir_mod * end2in + ((1-dir_mod) * end1in)

    # Get the two vectors representing our line segments.
    vec1 = end2 - end1
    vec2 = p2 - p1

    # Take the norm of each vector, and find the relative scale.
    norm1 = torch.linalg.norm(vec1)
    norm2 = torch.linalg.norm(vec2)
    scale = norm2/norm1

    #print(norm1, norm2)
    # Get rescaled versions of each vector.
    # NOT SAFE FOR VECTORS OF LENGTH 0.
    xa, ya = vec1/norm1
    xb, yb = vec2/norm2

    # Define the rotation matrix that roates xa,ya to be the
    # same direction as xb,yb.
    Rms = torch.tensor(
        [[xa*xb + ya*yb, xb*ya - xa*yb],
         [xa*yb - xb*ya, xa*xb + ya*yb]],
         device=scale.device
    )

    # Add in the scaling factor, and pad to size so that we have
    # a valid affine transformation matrix.
    Rm = torch.cat([
                torch.cat([scale*Rms, 0*p1.unsqueeze(1)],axis=1),
                torch.tensor([[0,0,1]],device=scale.device)
            ], axis=0)

    # Multiply that transformation matrix with the matrices representing translation
    # from p1 to the origin, and from the origin to end1.
    tmat = torch.matmul(
            torch.cat([
                torch.cat([torch.eye(2,device=scale.device), p1.unsqueeze(1)],axis=1),
                torch.tensor([[0,0,1]],device=scale.device)
            ], axis=0),
            torch.matmul(
                Rm,
                torch.cat([
                    torch.cat([torch.eye(2,device=scale.device), -1.0*end1.unsqueeze(1)],axis=1),
                    torch.tensor([[0,0,1]],device=scale.device)
                ], axis=0)


            )

    )
    # Return the translation matrix and the scaling factor.
    #print(tmat)
    return scale, tmat
