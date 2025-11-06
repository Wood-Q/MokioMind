import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim:int,eps:float=1e-5):
        super().__init__()
        self.dim=dim
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))
    
    def _norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
    
    def forward(self,x):
        return self.weight*self._norm(x.float()).typed_as(x)