import torch
import torch.nn as nn

aa = torch.randn(3, 512, 32, 3)
aa = aa.permute(0, 3, 2, 1)
net = nn.Conv2d(3, 64, 1)

bb = net(aa)

pass