from data import dataset
import torch
import argparse
import model
import trainer
import arary_tool as at
import configs


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
    fastRCNN = model.FastCNN().to(device)
    fastRCNNTrainer = trainer.FasterRCNNTrainer(fastRCNN).to(device)
    total_step = len(train_loader)
    for epoch in range(configs.epoch):
        for i, [images, bbox, label, scale] in enumerate(train_loader):
            scale = at.scalar(scale)
            img, bbox, label = images.to(device), bbox.cuda(), label.cuda()
            # fastRCNNTrainer(img, bbox, label, scale)
            loss = fastRCNNTrainer.train_step(img, bbox, label, scale)
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {}'
                      .format(epoch + 1, args.epoch, i + 1, total_step, loss.total_loss.item()))


if __name__ == '__main__':
    args = parse_args()
    main(args)
