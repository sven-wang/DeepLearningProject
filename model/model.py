import numpy as np
import torch.utils.data
from torch.utils.data.dataset import Dataset
import torch
import numpy as np
import os
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

"""
Problems:
0. How to represent each data? Transform the full length of wav into a matrix or just part of it or maybe fix length of it?
1. How to make sure that outputs from CNN are all same sized?
2. How to select triplets (Anchor, Pos and Neg)?
3. How to iterate three dataloaders?
"""


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN).__init__()

        self.conv1 = torch.nn.Conv1d(40, 96, 3, 1, 1)
        self.relu1 = torch.nn.LeakyReLU()
        self.conv2 = torch.nn.Conv1d(96, 96, 3, 1, 1)
        self.relu2 = torch.nn.LeakyReLU()
        self.conv3 = torch.nn.Conv1d(96, 192, 3, 1, 1)
        self.relu3 = torch.nn.LeakyReLU()

        # self.dropout2 = torch.nn.Dropout(0.5)
        self.conv4 = torch.nn.Conv1d(192, 192, 3, 1, 1)
        self.relu4 = torch.nn.LeakyReLU()
        self.conv5 = torch.nn.Conv1d(192, 192, 3, 1, 1)
        self.relu5 = torch.nn.LeakyReLU()
        self.conv6 = torch.nn.Conv1d(192, 192, 3, 1, 1)
        self.relu6 = torch.nn.LeakyReLU()

        # self.dropout3 = torch.nn.Dropout(0.5)
        self.conv7 = torch.nn.Conv1d(192, 384, 3, 1, 1)
        self.relu7 = torch.nn.LeakyReLU()
        self.conv8 = torch.nn.Conv1d(384, 384, 3, 1, 1)
        self.relu8 = torch.nn.LeakyReLU()
        self.conv9 = torch.nn.Conv1d(384, 384, 3, 1, 1)
        self.relu9 = torch.nn.LeakyReLU()

        self.conv10 = torch.nn.Conv1d(384, 384, 3, 1, 1)
        self.relu10 = torch.nn.LeakyReLU()
        self.conv11 = torch.nn.Conv1d(384, 384, 3, 1, 1)
        self.relu11 = torch.nn.LeakyReLU()
        self.conv12 = torch.nn.Conv1d(384, 192, 3, 1, 1)
        self.relu12 = torch.nn.LeakyReLU()

        self.conv13 = torch.nn.Conv1d(192, 192, 3, 1, 1)
        self.relu13 = torch.nn.LeakyReLU()
        self.conv14 = torch.nn.Conv1d(192, 192, 3, 1, 1)
        self.relu14 = torch.nn.LeakyReLU()
        self.conv15 = torch.nn.Conv1d(192, 192, 3, 1, 1)
        self.relu15 = torch.nn.LeakyReLU()

        self.conv16 = torch.nn.Conv1d(192, 96, 1, 1, 0)
        self.relu16 = torch.nn.LeakyReLU()
        self.conv17 = torch.nn.Conv1d(96, 96, 1, 1, 0)
        self.relu17 = torch.nn.LeakyReLU()
        self.conv18 = torch.nn.Conv1d(96, 46, 1, 1, 0)
        self.relu18 = torch.nn.LeakyReLU()

        self.bn1 = torch.nn.BatchNorm1d(96)
        self.bn2 = torch.nn.BatchNorm1d(96)
        self.bn3 = torch.nn.BatchNorm1d(192)
        self.bn4 = torch.nn.BatchNorm1d(192)
        self.bn5 = torch.nn.BatchNorm1d(192)
        self.bn6 = torch.nn.BatchNorm1d(192)
        self.bn7 = torch.nn.BatchNorm1d(384)
        self.bn8 = torch.nn.BatchNorm1d(384)
        self.bn9 = torch.nn.BatchNorm1d(384)
        self.bn10 = torch.nn.BatchNorm1d(384)
        self.bn11 = torch.nn.BatchNorm1d(384)
        self.bn12 = torch.nn.BatchNorm1d(192)
        self.bn13 = torch.nn.BatchNorm1d(192)
        self.bn14 = torch.nn.BatchNorm1d(192)
        self.bn15 = torch.nn.BatchNorm1d(192)
        self.bn16 = torch.nn.BatchNorm1d(96)
        self.bn17 = torch.nn.BatchNorm1d(96)
        self.bn18 = torch.nn.BatchNorm1d(46)

        self.pooling = AvgPool()

        # MLP layer
        # TODO: change linear layer dimension. Pad input data at the begging or fix in put data dimension?
        self.linear = torch.nn.Linear(28 * 28, 500)
        self.mlp = torch.nn.LeakyReLU()

    def forward(self, input_val, onehot, num_frames, pre_train=True):
        input_val = input_val.transpose_(1, 2)

        conv1 = self.conv1(input_val)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = self.relu2(bn2)
        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)
        part_one = self.relu3(bn3)

        part_two = self.relu6(self.bn6(self.conv6(self.relu5(
            self.bn5(self.conv5(self.relu4(self.bn4(self.conv4(part_one)))))))))

        part_three = self.relu9(self.bn9(self.conv9(
            self.relu8(self.bn8(self.conv8(self.relu7(self.bn7(self.conv7(part_two)))))))))

        part_four = self.relu12(self.bn12(self.conv12
            (self.relu11(self.bn11(self.conv11(self.relu10(self.bn10(self.conv10(part_three)))))))))

        part_five = self.relu15(self.bn15(self.conv15
            (self.relu14(self.bn14(self.conv14(self.relu13(self.bn13(self.conv13(part_four)))))))))

        part_six = self.relu18(self.bn18(self.conv18
            (self.relu17(self.bn17(self.conv17(self.relu16(self.bn16(self.conv16(part_five)))))))))

        conv_out = part_six

        pool = self.pooling(conv_out, onehot, num_frames)

        if pre_train:
            return self.mlp(self.linear(pool))
        else:
            return conv_out


class AvgPool(torch.nn.Module):

    def forward(self, conv_out, onehot, num_frames):
        onehot = onehot.transpose_(1, 2).squeeze()
        conv_out = conv_out.transpose_(1, 2).squeeze()
        num_frames = num_frames.transpose_(0, 1)
        res = torch.mm(onehot, conv_out)
        res = res / num_frames
        return res


class MyDataset(Dataset):
    def __init__(self, dir):
        self.data_files = os.listdir(dir)  # loads a list of files in __init__
        # TODO: sort data files to make sure training data are aligned?

    def __getindex__(self, idx):
        return load_file(self.data_files[idx])

    def __len__(self):
        return len(self.data_files)


def load_file(filename):
    """
    TODO: load data matrix and return matrix and its label
    open file, load data, and load label
    """
    return


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


def collate():
    return


if __name__ == "__main__":
    batch_size = 1
    dataset1 = MyDataset("DIR")
    loader1 = torch.utils.data.DataLoader(dataset1, shuffle=False, batch_size=batch_size, collate_fn=collate)

    # TODO: iterate data
    loss_fn = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    # loss = loss_fn(input1, input2, input3) #anchor, positive, negative
    model = CNN()
