import torch
import torch.nn as nn
from torch.utils.data import Dataset
import math
from torch.autograd import Function
import numpy as np
import os
import csv
import pickle


def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()


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
        label_person_map = {}
        for data_file in self.data_files:
            person = data_file.split("-")[0]
            if person not in self.label_dict:
                self.label_dict[person] = cnt
                label_person_map[cnt] = person
                cnt += 1
        self.total_labels = len(self.label_dict)

        with open("person2label_map.pickle", 'wb') as handle:
            pickle.dump(self.label_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("label2person_map.pickle", 'wb') as handle:
            pickle.dump(label_person_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('number of classes', self.total_labels)
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


class AvgPool_2d(torch.nn.Module):
    def forward(self, conv_out):
        res = torch.mean(conv_out, dim=2).squeeze(dim=-1)

        return res


class DeepSpeakerModel(nn.Module):
    def __init__(self, num_classes):
        super(DeepSpeakerModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.LeakyReLU()

        self.conv5 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.LeakyReLU()

        self.avgpool = AvgPool_2d()
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        # Input data dimension is (batch_size, 1, time_step, feature_dim)
        x = torch.unsqueeze(x, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.avgpool(x)
        feat_res = x
        x = self.fc(x)

        return x, feat_res
