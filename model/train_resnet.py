from resnet import *
import os
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np


def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()


def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


def main(num_of_classes):

    batch_size = 1
    lr = 0.001
    epochs = 20

    # Init model
    model = DeepSpeakerModel(num_of_classes)

    # Load dataset
    dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(os.path.dirname(dir), "data/")  # directory of single training instances
    pretrain_dataset = MyDataset(dir)

    # Currently batch size set to 1. Padding required for >1 batch size.
    pretrain_loader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss(size_average=False)
    #optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.001)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = StepLR(optim, step_size=3, gamma=0.8)

    if torch.cuda.is_available():
        # Move the network and the optimizer to the GPU
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    for epoch in range(epochs):
        print("Epoch: " + str(epoch))
        losses = []
        counter = 1
        scheduler.step()
        for (input_val, label) in pretrain_loader:
            optim.zero_grad()

            prediction = model(to_variable(input_val))
            print(prediction)

            label = label.transpose_(0, 1).long().resize_(batch_size)
            print(label)
            loss = loss_fn(prediction, to_variable(label))
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            optim.step()

            if counter % 1 == 0:
                print(loss)
            counter += 1

        print("Epoch {} Loss: {:.4f}".format(epoch, np.asscalar(np.mean(losses))))


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
    classes = get_class_num()
    main(classes)
