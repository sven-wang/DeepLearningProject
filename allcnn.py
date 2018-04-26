import numpy as np
import torch.utils.data
from torch.utils.data.dataset import Dataset
import torch
import numpy as np
import os
import sys
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_squared_error
from math import sqrt
from torch import nn


"""
Problems:
0. How to represent each data? Transform the full length of wav into a matrix or just part of it or maybe fix length of it?
try different lengths.
2. How to select triplets (Anchor, Pos and Neg)?
randome choice.
3. How to iterate three dataloaders?
triplet loader

pretrain loss fn: weighted cross entropy
"""

# Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0)
def all_cnn():
    return nn.Sequential(
        nn.Conv1d(40, 96, 3, 1, 1),
        nn.BatchNorm1d(96),
        nn.LeakyReLU(),
        nn.Conv1d(96, 96, 3, 2, 1),
        nn.BatchNorm1d(96),
        nn.LeakyReLU(),
        nn.Conv1d(96, 96, 3, 1, 1),
        nn.BatchNorm1d(96),
        nn.LeakyReLU(),

        nn.Conv1d(96, 192, 3, 2, 1),
        nn.BatchNorm1d(192),
        nn.LeakyReLU(),
        nn.Conv1d(192, 192, 3, 1, 1),
        nn.BatchNorm1d(192),
        nn.LeakyReLU(),
        nn.Conv1d(192, 384, 3, 1, 1),
        nn.BatchNorm1d(384),
        nn.LeakyReLU(),

        nn.Conv1d(384, 384, 3, 2, 1),
        nn.BatchNorm1d(384),
        nn.LeakyReLU(),
        nn.Conv1d(384, 384, 3, 1, 1),
        nn.BatchNorm1d(384),
        nn.LeakyReLU(),
        nn.Conv1d(384, 384, 3, 1, 1),
        nn.BatchNorm1d(384),
        nn.LeakyReLU(),

        nn.Conv1d(384, 384, 3, 2, 1),
        nn.BatchNorm1d(384),
        nn.LeakyReLU(),
        nn.Conv1d(384, 384, 3, 1, 1),
        nn.BatchNorm1d(384),
        nn.LeakyReLU(),
        nn.Conv1d(384, 384, 3, 1, 1),
        nn.BatchNorm1d(384),
        nn.LeakyReLU(),

        nn.Conv1d(384, 384, 3, 2, 1),
        nn.BatchNorm1d(384),
        nn.LeakyReLU(),
        nn.Conv1d(384, 384, 3, 1, 1),
        nn.BatchNorm1d(384),
        nn.LeakyReLU(),
        nn.Conv1d(384, 384, 3, 1, 1),
        nn.BatchNorm1d(384),
        nn.LeakyReLU(),

    )


class CNN(torch.nn.Module):

    def __init__(self, num_of_class):
        super(CNN, self).__init__()
        self.num_of_class = num_of_class

        self.cnn = all_cnn()

        self.pooling = AvgPool()
        # MLP layer
        self.linear = torch.nn.Linear(384, self.num_of_class)
        self.mlp = torch.nn.LeakyReLU()

    def forward(self, input_val, pre_train=True):
        input_val = input_val.transpose_(1, 2)

        # print(input_val.size())
        conv_out = self.cnn(input_val)     # TODO: change back to part_six

        # print(conv_out.size())

        # input, output size
        # torch.Size([1, 40, 30155])
        # torch.Size([1, 384, 943])

        pool = self.pooling(conv_out)

        if pre_train:
            return self.mlp(self.linear(pool))
        else:
            return conv_out


class AvgPool(torch.nn.Module):

    def forward(self, conv_out):
        res = torch.mean(conv_out, dim=2)
        return res


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



def init_xavier(m):
    if type(m) == torch.nn.Conv1d:
        torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.constant(m.bias, 0.0)


def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()


def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


def expand_labels(labels):
    b = np.zeros((labels.shape[0], 46))
    b[np.arange(labels.shape[0]), labels] = 1
    return b


if __name__ == "__main__":

    # classes = get_class_num()
    classes = 351
    prev_state = None
    if len(sys.argv) == 2:
        prev_state = sys.argv[1]

    batch_size = 1
    lr = 0.001
    epochs = 100
    num_of_class = 351    # TODO: change to REAL number of classes
    pretrain = True
    datadir = 'train2008_features/'

    batch_size = 1

    # Init model
    model = CNN(num_of_class)

    # if load previous state
    if prev_state:
        model.load_state_dict(torch.load(prev_state))

    # Load dataset
    # dir = os.path.dirname(os.path.abspath(__file__))
    # dir = os.path.join(os.path.dirname(dir), "data/")  # directory of single training instances

    pretrain_dataset = MyDataset('train.txt', datadir)
    dev_dataset = MyDataset('dev.txt', datadir)

    # Currently batch size set to 1. Padding required for >1 batch size.
    pretrain_loader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss(size_average=False)
    # optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.001)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # scheduler = StepLR(optim, step_size=3, gamma=0.8)

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
        for (input_val, label) in pretrain_loader:
            optim.zero_grad()

            prediction= model(to_variable(input_val))
            # print(prediction)

            label = label.transpose_(0, 1).long().resize_(batch_size)
            # print(label)
            loss = loss_fn(prediction, to_variable(label))
            loss.backward()
            lossnp = loss.data.cpu().numpy()
            losses.append(lossnp)
            optim.step()

            if counter % interval == 0:
                print('Train Loss: %.2f  Progress: %d%%' % (lossnp[0], counter * 100 / total))
            counter += 1

        print("Epoch {} Loss: {:.4f}".format(epoch, np.asscalar(np.mean(losses))))

        # validation
        count_match = 0
        losses = []
        rmse_sum = 0.0
        rmse_count = 0
        model.eval()
        for (input_val, label) in dev_loader:
            prediction = model(to_variable(input_val))

            label = label.transpose_(0, 1).long().resize_(batch_size)
            loss = loss_fn(prediction, to_variable(label))
            lossnp = loss.data.cpu().numpy()
            losses.append(lossnp)

            prediction2 = prediction.data.cpu().numpy()
            prediction3 = np.argmax(prediction2, axis=1)

            # print (prediction2.shape)

            label_array = np.zeros((batch_size, prediction2.shape[1]))
            label_array[0][label.numpy()] = 1

            # print(label.numpy())
            # print(label_array[0])
            # print(prediction2[0])
            rmse = sqrt(mean_squared_error(label_array[0], prediction2[0]))
            # print(rmse)
            rmse_sum += rmse
            rmse_count += 1

            if prediction3 == label.numpy():
                count_match += 1

        dev_loss = np.asscalar(np.mean(losses))
        if dev_loss < best_loss:
            torch.save(model.state_dict(), 'best_state_2')
            best_loss = dev_loss

        print("Accuracy: " + str(count_match) + " matches!")
        print("RMSE: " + str(rmse_sum / rmse_count))
        print("Epoch {} Validation Loss: {:.4f}".format(epoch, dev_loss))
