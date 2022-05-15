import torch.utils.data as data
import provider
import settings
import os
import numpy as np


class modelNet40(data.Dataset):
    def __init__(self):
        self.tranfile = provider.getDataFiles(
            os.path.join(settings.BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
        self.train = []
        self.label = []
        train_file_idxs = np.arange(0, len(self.tranfile))
        for ele in range(len(self.tranfile)):
            current_data, current_label = provider.loadDataFile(self.tranfile[train_file_idxs[ele]])
            for ele_2 in range(0, current_data.shape[0]):
                self.train.append(current_data[ele_2, :, :])
                self.label.append(current_label[ele_2, :])

    def __getitem__(self, index):
        return self.train[index], self.label[index]

    def __len__(self):
        return len(self.train)
