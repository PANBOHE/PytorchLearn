#多维输入

import numpy as np
import torch
import matplotlib.pyplot as plt

#1.数据准备
#dataset
#torch自带有模型和数据，可以自己下载，也可以直接用代码下载
#
database = np.loadtxt('yourdata', delimiter = ',', dtype = np.float32)

#输入为多维矩阵，，第一个：代表所有行，第二：-1代表取所有列，但是不取最后一列（最后一列是Y值）
x_data = torch.from_numpy(database[:, :-1])
y_data = torch.from_numpy(database[:,[-1]])


#2.模型设计
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        #注意,nn.Sigmoid() 和torch.Function.Sigmoid()的区别
        self.sigmoid = torch.nn.Sigmoid()  #这里是激活函数



    def forward(self, x):
        #这里其实是o1,o2,o3
        #x= o3
        #但是用了同一个值来传递
        x = self.sigmoid(self.linear1( x ))
        x = self.sigmoid(self.linear2( x ))
        x = self.sigmoid(self.linear3( x ))
        
        return x # y_hat



model = Model()

criterion = torch.nn.BCELoss(reduction = 'mean')
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1 )

epoch_list = []
loss_list = []

for epoch in range(100):
    y_pred = model(x_data)  #这里没有用mini_batch ,直接用的源数据，到后面要用dataloader
    loss = criterion(y_pred, y_data)

    print(f" ===== Train Info ======: Now epoch is {epoch}, the loss inf is {loss.item()}")

    epoch_list.append(epoch)
    loss_list.append(loss.item())

    optimizer.zero_grad() #reset gradient
    loss.backward() #反向传播，计算当前梯度
    #这里loss函数在实例化的时候，应该已经自动调用了backward()
    #这里为什么要再调一次
    optimizer.step() #根据梯度更新网络参数




plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
    