import torch
t1 = torch.Tensor([1, 2, 3])
t2 = t1.unsqueeze(0)
print(t2)