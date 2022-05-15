import argparse
import torch
import pointnet_cls
import datasetBuilder
import numpy as np
import pointnetplus_cls


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
    # model = pointnet_cls.pointNet(3, args.num_class).to(device)
    # criterion = pointnet_cls.get_loss()
    model = pointnetplus_cls.PointNetPlus(args.num_class).to(device)
    criterion =pointnetplus_cls.get_loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.decay_rate
    )

    # load data
    train_dataset = datasetBuilder.modelNet40()
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    total_step = len(train_loader)
    for epoch in range(args.epoch):
        mean_correct = []
        for i, (pcls, labels) in enumerate(train_loader):
            # load batch
            pcls = pcls.to(device)
            labels = labels.to(device).view(-1)

            #input into model
            outputs, trans = model(pcls)
            loss = criterion(outputs, labels.long())

            # define acc
            pred_choice = outputs.max(1)[1]
            correct = pred_choice.eq(labels.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(pcls.shape[0]))

            # gradient op
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, args.epoch, i + 1, total_step, loss.item()))
                train_instance_acc = np.mean(mean_correct)
                print('Train Instance Accuracy: %f' % train_instance_acc)


if __name__ == '__main__':
    args = parse_args()
    main(args)
