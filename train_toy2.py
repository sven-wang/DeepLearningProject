from toy_2d import *
import os
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torch.utils.data.dataset import Dataset
import sys
from train_tripletloss import DevDataset
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


def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()


def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return Variable(tensor)


def main(num_of_classes, datadir, prev_state, lr, epochs):
    batch_size = 128

    # Init model
    model = DeepSpeakerModel(num_of_classes)

    # if load previous state
    if prev_state:
        model.load_state_dict(torch.load(prev_state))

    # Load dataset
    pretrain_dataset = MyDataset('train.txt', datadir)

    # Currently batch size set to 1. Padding required for >1 batch size.
    pretrain_loader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    # optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.001)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)

    # scheduler = StepLR(optim, step_size=3, gamma=0.8)

    with open("enrol_file_map_sample.pickle", "rb") as handle:
        enrol_file_map = pickle.load(handle)

    with open("test_file_set_sample.pickle", "rb") as handle:
        test_file_set = pickle.load(handle)

    dev_dataset = DevDataset("trials.txt", "enrol_features/", "test_features/", enrol_file_map,
                             test_file_set)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=8, shuffle=False)

    if torch.cuda.is_available():
        # Move the network and the optimizer to the GPU
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    best_loss = 999

    for epoch in range(epochs):
        print("Epoch: " + str(epoch))
        losses = []
        counter = 0
        total = len(pretrain_dataset)
        interval = int(total / batch_size / 10)

        # scheduler.step()
        model.train()
        for (input_val, label) in pretrain_loader:
            optim.zero_grad()

            prediction, _ = model(to_variable(input_val))
            cur_batch_size = input_val.shape[0]
            label = label.transpose(0, 1).long().resize_(cur_batch_size)

            loss = loss_fn(prediction, to_variable(label))
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            optim.step()

            # Gradient clipping with maximum norm 0.25
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            if counter % interval == 0:
                print('Train Loss: %.2f  Progress: %d%%' % (np.asscalar(np.mean(losses)), counter * 100 * cur_batch_size / total))
            counter += 1

        print("Epoch {} Loss: {:.4f}".format(epoch, np.asscalar(np.mean(losses))))

        # for EER
        y_gold_pred = []
        y_pred_pred = []
        y_gold_feat = []
        y_pred_feat = []
        model.eval()
        for (data_a, data_p, label) in dev_dataloader:
            data_a, data_p = to_variable(data_a), to_variable(data_p)

            # compute output
            out_a_pred, out_a_feat = model(data_a)
            out_p_pred, out_p_feat = model(data_p)

            # record similarity and true label for both pairs
            np_a_pred = out_a_pred.data.cpu().numpy()
            np_p_pred = out_p_pred.data.cpu().numpy()
            np_a_feat = out_a_feat.data.cpu().numpy()
            np_p_feat = out_p_feat.data.cpu().numpy()

            for i in range(np_a_pred.shape[0]):
                y_gold_pred.append(int(label.numpy()[0]))
                y_pred_pred.append(1 - cosine(np_a_pred[i], np_p_pred[i]))
                y_gold_feat.append(int(label.numpy()[0]))
                y_pred_feat.append(1 - cosine(np_a_feat[i], np_p_feat[i]))

        print('Validation EER (using prediction output):')
        print(eer(y_gold_pred, y_pred_pred))
        print('Validation EER (using feature):')
        print(eer(y_gold_feat, y_pred_feat))

        torch.save(model.state_dict(), 'best_state_toy')


def get_class_num():
    dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(os.path.dirname(dir), "data/")  # directory of single training instances
    num_of_classes = set()

    for filename in os.listdir(dir):
        if filename.endswith(".npy"):
            person = filename.split("-")[0]
            num_of_classes.add(person)
    return len(num_of_classes)


if __name__ == "__main__":
    # classes = get_class_num()
    classes = 2382
    prev_state = None
    if len(sys.argv) == 2:
        prev_state = sys.argv[1]
    main(num_of_classes=classes, datadir='./new_features_2000/', prev_state=prev_state, lr=0.001, epochs=1000)
