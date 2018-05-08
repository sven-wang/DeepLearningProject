from toy_2d import *
import os
import torch
import numpy as np
from scipy.spatial.distance import cosine
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


class TADataset(Dataset):
    def __init__(self, filedir):
        self.dir = filedir
        self.pairs = []
        with open(filedir, 'r') as f:
            for line in f:
                speaker_file, test_file, label = line.strip().split()
                self.pairs.append((speaker_file, test_file, label))

    def __getitem__(self, item):
        speaker_file, test_file, label = self.pairs[item]

        print("#######################################################")
        person1 = speaker_file[0:-4].capitalize()
        person2 = test_file[0:-4].capitalize()
        print("TA 1: %s, TA 2: %s" % (person1, person2))

        enrol = np.load(speaker_file)  # eg. 32707 # todo: double check
        test = np.load(test_file)  # eg. tkwut_A
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
    # print('eer_threshold:', thresh)

    return eer


def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


def test():
    test_dataset = TADataset("tas.txt")  # todo: double check
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total = len(test_dataset)

    # Load Model
    model = DeepSpeakerModel(classes)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    correct = 0
    for (data_a, data_p, label) in test_dataloader:
        data_a, data_p = to_variable(data_a), to_variable(data_p)

        # compute output
        out_a, feat_a = model(data_a)
        out_p, feat_p = model(data_p)

        # record similarity and true label for both pairs
        np_a = feat_a.data.cpu().numpy()
        np_p = feat_p.data.cpu().numpy()

        similarity = 1 - cosine(np_a[0], np_p[0])
        if similarity > 0.7:
            print("I know it! Same instructor!")
            if label.numpy()[0] == 1:
                correct += 1
            else:
                print(similarity)
        else:
            print("Can't fool me! Not the same instructor!")
            if label.numpy()[0] == 0:
                correct += 1
            else:
                print(similarity)

    print("=== THE END ===")
    print("Accuracy: %f%%" % (correct / total * 100))


if __name__ == "__main__":
    batch_size = 1
    classes = 2382

    # todo: modify
    model_path = 'best_state_toy_epoch18'

    test()
