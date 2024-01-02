import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

input=torch.tensor([1,2,3],dtype=float)
target=torch.tensor([1,2,5],dtype=float)

input=torch.reshape(input,(1,1,1,3))
target=torch.reshape(target,(1,1,1,3))

loss1=L1Loss()
# MSE常用于回归问题（预测房价等）
loss2=MSELoss()
# CrossEntropyLoss常用于分类问题
loss3=CrossEntropyLoss()
result1=loss1(input,target)
result2=loss2(input,target)
x=torch.tensor([0.1,0.3,0.6])
y=torch.tensor([2])
x=torch.reshape(x,(1,3))
y=torch.reshape(y,(1,))
result3=loss3(x,y)
print(result1)
print(result2)
print(result3)