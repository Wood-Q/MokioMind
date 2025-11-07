import torch
v1=torch.tensor([1,2,3])
v2=torch.tensor([4,5,6])
result=torch.outer(v1,v2)
print(result)