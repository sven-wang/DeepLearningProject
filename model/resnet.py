import torch
import torch.nn as nn
from torch.utils.data import Dataset
import math
from torch.autograd import Function
import numpy as np
import os
import csv

"""
Changed all Conv2d to Conv1d, Batchnorm2d to Batchnorm1d. 
"""


def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()


class CosineSimilarity(Function):
    def __init__(self):
        super(CosineSimilarity, self).__init__()

    def forward(self, x1, x2):
        assert x1.size() == x2.size()

        res = torch.dot(to_tensor(x1), torch.transpose(to_tensor(x2), 0, 1))
        similarity = res.cpu().numpy()[0]

        return similarity


class PairwiseDistance(Function):
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)


class TripletMarginLoss(Function):
    """Triplet loss function.
    """
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)  # norm 2

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(anchor, positive)
        d_n = self.pdist.forward(anchor, negative)

        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss


class TripletDataset(Dataset):
    def __init__(self, filedir, feats_dir):
        self.dir = filedir
        self.feats_dir = feats_dir

        self.triplets = []
        with open(filedir, 'r') as csvfile:
            for line in csvfile:
                anchor_file, positive_file, negative_file = line.strip().split(",")
                self.triplets.append((anchor_file, positive_file, negative_file))

    def __getitem__(self, item):
        anchor_file, positive_file, negative_file = self.triplets[item]
        anchor = np.load(self.feats_dir + anchor_file)
        positive = np.load(self.feats_dir + positive_file)
        negative = np.load(self.feats_dir + negative_file)

        return to_tensor(anchor), to_tensor(positive), to_tensor(negative)

    def __len__(self):
        return len(self.triplets)


class MyDataset(Dataset):
    def __init__(self, txtfile, datadir):
        self.dir = datadir
        f = open(txtfile)
        self.data_files = f.readlines()  # loads a list of files in __init__
        # Select only numpy files
        # self.data_files = [file for file in self.data_files if file.endswith(".npy")]
        self.data_files = [file.strip() for file in self.data_files]

        # print(self.data_files)

        # Get total number of classes and save into a dictionary
        cnt = 0
        self.label_dict = {}
        for data_file in self.data_files:
            person = data_file.split("-")[0]
            if person not in self.label_dict:
                self.label_dict[person] = cnt
                cnt += 1
        self.total_labels = len(self.label_dict)

        print('loaded %s' % txtfile)

    def __getitem__(self, item):
        # Get training data
        filename = self.data_files[item]

        X = np.load(self.dir+filename)

        # Build data label one-hot vector
        person = filename.split("-")[0]
        idx = np.array([self.label_dict[person]])
        # Y = np.zeros([self.total_labels], dtype=float)
        # Y[idx] = 1
        return to_tensor(X), to_tensor(idx)

    def __len__(self):
        return len(self.data_files)


class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class AvgPool(torch.nn.Module):

    def forward(self, conv_out):

        res = torch.mean(conv_out, dim=2)
        return res


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)

        self.relu = ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class myResNet(nn.Module):

    def __init__(self, block, layers, num_classes):

        super(myResNet, self).__init__()

        self.relu = ReLU(inplace=True)
        self.inplanes = 64
        self.conv1 = nn.Conv1d(40, 64, kernel_size=5, stride=2, padding=2,bias=False)

        self.bn1 = nn.BatchNorm1d(64)

        self.layer1 = self._make_layer(block, 64, layers[0])

        self.inplanes = 128
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2,bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.inplanes = 256
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2,bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.inplanes = 512
        self.conv4 = nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2,bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.layer4 = self._make_layer(block, 512, layers[3])

        self.avgpool = AvgPool()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class DeepSpeakerModel(nn.Module):
    def __init__(self, num_classes):
        super(DeepSpeakerModel, self).__init__()

        self.model = myResNet(BasicBlock, [1, 1, 1, 1], num_classes)
        self.model.fc = nn.Linear(512, num_classes)

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = x.transpose_(1, 2)

        x = self.model.conv1(x)
        x = self.model.bn1(x)

        x = self.model.relu(x)
        x = self.model.layer1(x)

        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer2(x)

        x = self.model.conv3(x)
        x = self.model.bn3(x)
        x = self.model.relu(x)
        x = self.model.layer3(x)

        x = self.model.conv4(x)
        x = self.model.bn4(x)
        x = self.model.relu(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)

        feat_res = x
        x = self.model.fc(x)

        self.features = self.l2_norm(x)
        alpha=10
        self.features = self.features*alpha

        return self.features, feat_res

    def forward_classifier(self, x):
        features = self.forward(x)
        #res = self.model.classifier(features)
        return features
