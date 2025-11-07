import torch
import torch.nn as nn

dropout_layer = nn.Dropout(p=0.5)

t1=torch.Tensor([1,2,3])
t2=dropout_layer(t1)
# 这里Dropout丢弃了1，为了保持期望不变，将1和3扩大两倍
print(t2)