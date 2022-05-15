from data import dataset
import torch
import argparse
import model


def parse_args():
    parser = argparse.ArgumentParser("PointNet")
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument("--num_class", type=int, default=40)
    return parser.parse_args()


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_data = dataset.VocTrainDataset()
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=1,
                                               shuffle=True)
    fastRCNN = model.FastCNN()
    for i, [images, _, _, _] in enumerate(train_loader):
        b = images
        aa = fastRCNN(images)
        pass


if __name__ == '__main__':
    args = parse_args()
    main(args)
