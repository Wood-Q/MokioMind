import torch
import torch.nn as nn

layer = nn.Linear(in_features=3, out_features=5, bias=True)
t1 = torch.Tensor([1, 2, 3])  # shape: (3,)

t2 = torch.Tensor([[1, 2, 3]])  # shape: (1, 3)
# 这里应用的w和b是随机的，真实训练里会在optimizer上更新
output2 = layer(t2)             # shape: (1, 5)
print(output2)