from resnet_2d import *
import os
import torch
from sklearn.metrics import roc_curve, auc
import numpy as np
from scipy.spatial.distance import cosine


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


class DevDataset(Dataset):
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


def train():
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(999)

    train_dataset = TripletDataset("triplets.csv", "new_features/")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load Model
    model_path = os.path.join(os.path.dirname(__file__), 'best_state')
    model = DeepSpeakerModel(classes)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)

    with open("enrol_file_map_sample.pickle", "rb") as handle:
        enrol_file_map = pickle.load(handle)

    with open("test_file_set_sample.pickle", "rb") as handle:
        test_file_set = pickle.load(handle)

    dev_dataset = DevDataset("trials.txt", "enrol_features/", "test_features/", enrol_file_map, test_file_set)  # todo: double check
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    print(len(dev_dataset))

    losses = []
    total = len(train_dataset)
    for epoch in range(epochs):
        # for EER
        y_gold = []
        y_pred = []

        counter = 0
        model.train()
        for (data_a, data_p, data_n) in train_dataloader:
            data_a, data_p, data_n = to_variable(data_a), to_variable(data_p), to_variable(data_n)

            # compute output
            out_a, out_p, out_n = model(data_a)[0], model(data_p)[0], model(data_n)[0]  # vector after the fc layer

            triplet_loss = TripletMarginLoss(margin).forward(out_a, out_p, out_n)

            # record similarity and true label for both pairs
            np_a = out_a.data.cpu().numpy()
            np_p = out_p.data.cpu().numpy()
            np_n = out_n.data.cpu().numpy()

            for i in range(np_a.shape[0]):
                y_gold.append(1)
                y_pred.append(1 - cosine(np_a[i], np_p[i]))
                y_gold.append(0)
                y_pred.append(1 - cosine(np_a[i], np_n[i]))

            # compute gradient and update weights
            optimizer.zero_grad()
            triplet_loss.backward()
            losses.append(triplet_loss.data.cpu().numpy())
            optimizer.step()

            if counter % 200 == 0:
                print('Train Loss: %.2f  Progress: %d%%' % (np.asscalar(np.mean(losses)), counter * 100 * batch_size / total))
                print('EER:', eer(y_gold, y_pred))
            counter += 1

        print("Epoch {} Loss: {:.4f}".format(epoch, np.asscalar(np.mean(losses))))
        print('EER:', eer(y_gold, y_pred))

        # for EER
        y_gold = []
        y_pred = []
        model.eval()
        for (data_a, data_p, label) in dev_dataloader:
            data_a, data_p = to_variable(data_a), to_variable(data_p)

            # compute output
            out_a, out_p = model(data_a)[0], model(data_p)[0]  # vector after the fc layer

            # record similarity and true label for both pairs
            np_a = out_a.data.cpu().numpy()
            np_p = out_p.data.cpu().numpy()

            for i in range(np_a.shape[0]):
                y_gold.append(int(label.numpy()[0]))
                y_pred.append(1 - cosine(np_a[i], np_p[i]))

        print('Validation EER:')
        print(eer(y_gold, y_pred))

        torch.save(model.state_dict(), 'best_state_triplet')


if __name__ == "__main__":
    batch_size = 24
    lr = 0.0001
    epochs = 100
    classes = 834
    margin = 0.1  # the margin value for the triplet loss function (default: 1.0)

    train()
