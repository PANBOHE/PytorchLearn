import torch

x_data = torch.Tensor(
    [
        [1.0],
        [2.0],
        [3.0],
        [4.0]
    ]
)


y_data = torch.Tensor(
    [
        [2.0],
        [4.0],
        [6.0],
        [8.0]
    ]
)
z = torch.Tensor([[3.0],[2.0]])

#print(x_data)
print(z)
#模型定义

#都得继承torch.nn.Module
#torch自带backward函数
class LinearModel(torch.nn.Module):
    def __init__(self) -> None:
        #调用父类的构造
        super(LinearModel,self).__init__()
        #入参：函数nn.linear contain two member Tensors: weight and bias
        self.linear = torch.nn.Linear(1, 1)
    
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    

#实例化
# model 是 callable.
model = LinearModel()

#构造损失函数和优化器
### class torch.nn.MSELoss(size_average = True, reduce = True)
#求均值可以都求，或者都不求。
criterion = torch.nn.MSELoss(size_average = False)


#优化器，实例化

optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

#训练的过程
#
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss)

    #梯度归零
    optimizer.zero_grad()
    loss.backward()
    #updata作用，更新
    optimizer.step()



#output weight and bias
print("w = ", model.linear.weight.item())
print("b = ", model.linear.bias.item())

#test model
x_test = torch.Tensor([[8.0]])
y_test = model(x_test)
print('y_pred', y_test.data)
