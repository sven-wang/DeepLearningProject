from toy_2d import *
import os
import torch
import numpy as np
from scipy.spatial.distance import cosine
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import sys


class TestDataset(Dataset):
    def __init__(self, person1, person2):
        self.person1 = person1
        self.person2 = person2

    def __getitem__(self, item):

        enrol = np.load(self.person1)
        test = np.load(self.person2)

        return to_tensor(enrol), to_tensor(test)

    def __len__(self):
        return 1


def eer(y_gold, y_pred):
    # y = [1, 1, 0, 0]
    # y_pred = [0.5, 0.8, 0.5, 0.1]

    fpr, tpr, thresholds = roc_curve(y_gold, y_pred, pos_label=1)

    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    print('eer_threshold:', thresh)

    return eer


def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


def test(ta1, ta2):

    test_dataset = TestDataset(ta1, ta2)  # todo: double check
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load Model
    model = DeepSpeakerModel(classes)
    model.load_state_dict(torch.load(model_path))

    if torch.cuda.is_available():
        model = model.cuda()

    counter = 1
    model.eval()
    for (data_a, data_p) in test_dataloader:
        data_a, data_p = to_variable(data_a), to_variable(data_p)

        # compute output
        out_a, feat_a = model(data_a)
        out_p, feat_p = model(data_p)

        # record similarity and true label for both pairs
        np_a = out_a.data.cpu().numpy()
        np_p = out_p.data.cpu().numpy()
        similarity = (1 - cosine(np_a[0], np_p[0]))
        print('using output', similarity)

        np_a = feat_a.data.cpu().numpy()
        np_p = feat_p.data.cpu().numpy()
        similarity = (1 - cosine(np_a[0], np_p[0]))
        print('using feature', similarity)


if __name__ == "__main__":
    batch_size = 1
    classes = 2382

    # todo: modify
    model_path = 'best_state_toy'

    test(sys.argv[1], sys.argv[2])
