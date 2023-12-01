import torch
from torch import nn
from d2l import torch as d2l
# 自己写了一遍，主要是注意RELU和dropout的加入，至于别的就是卷积层和全连接层加入而已，批量默认的是1
net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3 ,padding=1), nn.ReLU(), # 卷积核默认kernelsize为3
    nn.Conv2d(384, 384, kernel_size=3 ,padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3 ,padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400 ,4096), nn.ReLU(),
    nn.Dropout(p=0.5), # 设置dropout选项，防止模型过大，过拟合了
    nn.Linear(4096 ,4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096 ,10)
)
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)
