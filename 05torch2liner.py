import torch


##已有数据 训练集
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
# z = torch.Tensor([[3.0],[2.0]])

# #print(x_data)
# print(z)
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

#w =  2.0104787349700928
#b =  -0.03080911375582218

torch.save(model, 'panbotest.pt') # 保存整个model的状态
# model=torch.load('mymodel.pth') # 这里已经不需要重构模型结构了，直接load就可以
# model.eval()


#test model
x_test = torch.Tensor([[8.0]])
y_test = model(x_test)
#预测值
print('y_pred', y_test.data)
#y_pred tensor([[16.0530]])

model=torch.load('panbotest.pt')
model.eval()


input_names = ['input']
output_names = ['output']
x = torch.randn(1,requires_grad=True)
torch.onnx.export(model,x, 'best.onnx',input_names=input_names, output_names=output_names, verbose='True')
