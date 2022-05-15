import torch.nn as nn
import settings
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


class Conv2dwithBatchRelu(nn.Module):
    def __init__(self, input_channel, out_channel, kernel_size):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(input_channel, out_channel, kernel_size),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class Conv2dwithBatch(nn.Module):
    def __init__(self, input_channel, out_channel, kernel_size):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(input_channel, out_channel, kernel_size),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class LinearwithBatchRelu(nn.Module):
    def __init__(self, input_channel, out_channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_channel, out_channel),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class TransformNet(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.input_channel = input_channel
        self.mlp_1 = nn.Sequential(
            Conv2dwithBatchRelu(input_channel, 64, 1),
            Conv2dwithBatchRelu(64, 128, 1),
            Conv2dwithBatchRelu(128, 1024, 1)
        )

        self.mlp_2 = nn.Sequential(
            LinearwithBatchRelu(1024, 512),
            LinearwithBatchRelu(512, 256)
        )

        self.mlp_3 = nn.Linear(256, input_channel * input_channel)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.mlp_1(x)
        ## max in the dimension of Numpoints
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        ##reduce feature
        x = self.mlp_2(x)
        x = self.mlp_3(x)

        ## to tranform
        iden = Variable(torch.from_numpy(np.eye(self.input_channel).flatten().astype(np.float32))).view(1,
                                                                                                        self.input_channel * self.input_channel).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.input_channel, self.input_channel)
        return x


class pointNet(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        # firstly transform
        self.transNet_1 = TransformNet(input_channel)
        # then to 64 dimension
        self.mlp_1 = nn.Sequential(
            Conv2dwithBatchRelu(input_channel, 64, 1)
        )
        # second transform
        self.transNet_2 = TransformNet(64)

        # to 1024 dimensions
        self.mlp_2 = nn.Sequential(
            Conv2dwithBatchRelu(64, 128, 1),
            Conv2dwithBatch(128, 1024, 1)
        )

        # full connected for classification
        self.fc1 = LinearwithBatchRelu(1024, 512)
        self.dp = nn.Dropout(0.7)
        self.fc2 = LinearwithBatchRelu(512, 256)
        self.fc3 = nn.Sequential(
            nn.Linear(256, output_channel),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # firstly transform
        trans = self.transNet_1(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)

        # then to 64 dimensions
        x = self.mlp_1(x)

        # 64 dimensions transform
        trans = self.transNet_2(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)

        # to 1024 dimensions
        # todo
        x = self.mlp_2(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # full connected for classification
        x = self.dp(self.fc1(x))
        x = self.dp(self.fc2(x))
        x = self.fc3(x)
        return x, trans


class get_loss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
