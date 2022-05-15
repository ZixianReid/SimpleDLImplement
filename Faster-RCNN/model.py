from torchvision.models import vgg16
from torch.utils.model_zoo import load_url
import torch
import configs
import torch.nn as nn
import bbox_tools
import numpy as np
import torch.nn.functional as F


def getVGG16():
    model = vgg16(False)
    model.load_state_dict(torch.load(configs.vgg16_pretrain_path))

    classifier = model.classifier
    classifier = list(classifier)
    del classifier[5]
    del classifier[2]
    classifier = nn.Sequential(*classifier)

    features = list(model.features)[:30]
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    return nn.Sequential(*features), classifier


class RegionProposalNetwork(nn.Module):
    def __init__(self, feat_stride=16, in_channels=512, mid_channels=512):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = bbox_tools.generate_anchor_base()
        self.feat_stride = feat_stride
        n_anchor = self.anchor_base.shape[0]
        self.cov1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.score = nn.Conv2d(mid_channels, n_anchor * 2,  kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))


    def forward(self, x, img_size, scale=1.):
        b, _, hh, ww = x.shape
        anchor = bbox_tools.enumerate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)
        n_anchor = anchor.shape[0] // (hh * ww)
        h = F.relu(self.conv1(x))
        # loc
        rpn_locs = self.loc(h)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(b, -1, 4)
        # socre
        rpn_score = self.score(h)
        rpn_scores = rpn_score.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_scores.view(b, hh, ww, n_anchor, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(b, -1)
        rpn_scores = rpn_scores.view(b, -1, 2)
        for i in range(b):
            pass


class FastCNN(nn.Module):
    def __init__(self):
        super(FastCNN, self).__init__()
        self.extractor, self.classifier = getVGG16()

    def forward(self, x):
        image_size = x.shape[2:]
        h = self.extractor(x)
        print(self.extractor.eval())
        return h


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = vgg16(False).to(device)
    aa = torch.load(configs.vgg16_pretrain_path)
    model.load_state_dict(torch.load(configs.vgg16_pretrain_path))
    print(model.eval())
    bb = model.classifier
    print(bb.eval())
