import torch

x=torch.tensor([1, 2, 3, 4, 5])
y=torch.tensor([10, 20, 30, 40, 50])

condition=(x>3)

result=torch.where(condition,x,y)

print(result)  # Output: tensor([10, 20, 30,  4,  5])