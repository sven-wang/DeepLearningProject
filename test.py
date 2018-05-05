from resnet_2d_small import *
import os
import torch
from sklearn.metrics import roc_curve, auc
import numpy as np
from scipy.spatial.distance import cosine


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

    fpr, tpr, threshold = roc_curve(y_gold, y_pred, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    print('eer_threshold:', eer_threshold)

    return EER


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
    y_gold = []
    y_pred = []

    counter = 1
    model.train()
    for (data_a, data_p, label) in test_dataloader:
        data_a, data_p = to_variable(data_a), to_variable(data_p)

        # compute output
        out_a, out_p = model(data_a)[0], model(data_p)[0]  # vector before the fc layer

        # record similarity and true label for both pairs
        np_a = out_a.data.cpu().numpy()
        np_p = out_p.data.cpu().numpy()

        for i in range(np_a.shape[0]):
            y_gold.append(int(label.numpy()[0]))
            y_pred.append(1 - cosine(np_a[i], np_p[i]))

        if counter % 3000 == 0:
            print('EER:', eer(y_gold, y_pred))
        counter += 1

    print('EER:', eer(y_gold, y_pred))


if __name__ == "__main__":
    batch_size = 1
    classes = 1303

    # todo: modify
    model_path = './experiments/triplet_loss/best_state_triplet_small'

    test()
