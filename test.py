from toy_2d import *
import os
import torch
import numpy as np
from scipy.spatial.distance import cosine
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


class TestDataset(Dataset):
    def __init__(self, filedir, enroldir, testdir, enrol_map, test_set):
        self.dir = filedir
        self.enrol_dir = enroldir
        self.test_dir = testdir
        self.enrol_map = enrol_map
        self.test_set = test_set

        self.pairs = []
        with open(filedir, 'r') as f:
            for line in f:
                speaker_file, test_file, label = line.strip().split()
                if speaker_file in self.enrol_map and test_file in self.test_set:
                    self.pairs.append((self.enrol_map[speaker_file], test_file + "-vad.npy", label))

    def __getitem__(self, item):
        speaker_file, test_file, label = self.pairs[item]

        enrol = np.load(self.enrol_dir + speaker_file)  # eg. 32707 # todo: double check
        test = np.load(self.test_dir + test_file)  # eg. tkwut_A
        label = np.ones(1) if label == 'target' else np.zeros(1)

        return to_tensor(enrol), to_tensor(test), to_tensor(label)

    def __len__(self):
        return len(self.pairs)


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


def test():
    with open("enrol_file_map.pickle", "rb") as handle:
        enrol_file_map = pickle.load(handle)

    with open("test_file_set.pickle", "rb") as handle:
        test_file_set = pickle.load(handle)

    test_dataset = TestDataset("trials.txt", "enrol_features/", "test_features/", enrol_file_map, test_file_set)  # todo: double check
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(len(test_dataset))

    # Load Model
    model = DeepSpeakerModel(classes)
    model.load_state_dict(torch.load(model_path))

    if torch.cuda.is_available():
        model = model.cuda()

    # for EER
    y_gold_out = []
    y_pred_out = []
    y_gold_feat = []
    y_pred_feat = []

    counter = 1
    model.eval()
    for (data_a, data_p, label) in test_dataloader:
        data_a, data_p = to_variable(data_a), to_variable(data_p)

        # compute output
        out_a, feat_a = model(data_a)
        out_p, feat_p = model(data_p)

        # record similarity and true label for both pairs
        np_a = out_a.data.cpu().numpy()
        np_p = out_p.data.cpu().numpy()
        for i in range(np_a.shape[0]):
            y_gold_out.append(int(label.numpy()[0]))
            y_pred_out.append(1 - cosine(np_a[i], np_p[i]))

        # record similarity and true label for both pairs
        np_a = feat_a.data.cpu().numpy()
        np_p = feat_p.data.cpu().numpy()

        for i in range(np_a.shape[0]):
            y_gold_feat.append(int(label.numpy()[0]))
            y_pred_feat.append(1 - cosine(np_a[i], np_p[i]))

        if counter % 300 == 0:
            print('Feature EER:', eer(y_gold_feat, y_pred_feat))
            print('Output EER:', eer(y_gold_out, y_pred_out))
        counter += 1

    print("=== THE END ===")
    print('Feature EER:', eer(y_gold_feat, y_pred_feat))
    print('Output EER:', eer(y_gold_out, y_pred_out))


if __name__ == "__main__":
    batch_size = 16
    classes = 2382

    # todo: modify
    model_path = 'best_state_toy'

    test()
