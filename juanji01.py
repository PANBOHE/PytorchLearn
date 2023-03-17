#创建一个卷积层
import torch
in_channels, out_channels = 5, 10
width, height = 100, 100
kernel_size = 3
bathc_size = 1


input = torch.randn(
    bathc_size, #B
    in_channels, #N
    width,     #W
    height      #H

)

conv_layer = torch.nn.Conv2d(
    in_channels, #N
    out_channels, #M
    kernel_size = kernel_size #3x3 卷积核大小

)

output = conv_layer(input)

print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)