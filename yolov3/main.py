from dataY import dataset
import torch



def transferbbox(bbox):
    output = torch.zeros(bbox.shape)
    output[:, :, 0] = (bbox[:, :, 1] + bbox[:, :, 3]) / 2
    output[:, :, 1] = (bbox[:, :, 0] + bbox[:, :, 2]) / 2
    output[:, :, 2] = (bbox[:, :, 3] - bbox[:, :, 1])
    output[:, :, 3] = (bbox[:, :, 2] - bbox[:, :, 0])
    return output

train_data = dataset.VocTrainDataset()
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=1,
                                           shuffle=True)

train_data.__getitem__(1)
for i, [images, bbox, label] in enumerate(train_loader):
    print(bbox)
    print("----------")
    print(transferbbox(bbox))
