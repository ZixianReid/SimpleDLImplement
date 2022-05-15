import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import provider
import os
import settings
import pointnet_cls
import numpy as np
import torch.nn.functional as F
import pointnetplus_cls
import datasetBuilder


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_dataset = datasetBuilder.modelNet40()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=24,
                                           shuffle=True)

# train_file_idxs = np.arange(0, len(TRAIN_FILES))
# current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[0]])
# current_data = torch.tensor(current_data).to(device)

for i, (pcls, labels) in enumerate(train_loader):
    pcls = pcls.to(device)
    model = pointnetplus_cls.PointNetPlus(40).to(device)
    model(pcls)

# for ele in range(len(TRAIN_FILES)):
#     current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[ele]])
#     for ele_2 in range(0, current_data.shape[0]):
#         train = current_data[ele_2,:,:]
#         lable = current_label[ele_2, :]
#         pass

# input = torch.randn(24, 3, 1028, requires_grad=True)
# # input = input.permute(0, 2, 1)
# pred_choice = input.max(1)[1]

# target = torch.tensor([1, 0, 4])
# target = target.reshape(3,1)
# target = target.view(-1)
# aa = F.log_softmax(input)
# output = F.nll_loss(F.log_softmax(input), target)

