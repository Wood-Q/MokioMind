import torch

t1=torch.tensor([[1,2,3],[4,5,6]])
t2=torch.tensor([[7,8,9],[10,11,12]])
result=torch.cat((t1,t2),dim=0)
print(result)  

result2=torch.cat((t1,t2),dim=1)
print(result2)  