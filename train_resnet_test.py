from resnet import *
import os
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torch.utils.data.dataset import Dataset
from sklearn.metrics import mean_squared_error
from math import sqrt
import sys


def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()


def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


def main(num_of_classes, datadir, prev_state, lr, epochs):

    batch_size = 1

    # Init model
    model = DeepSpeakerModel(num_of_classes)

    # if load previous state
    if prev_state:
        model.load_state_dict(torch.load(prev_state))

    # Load dataset
    # dir = os.path.dirname(os.path.abspath(__file__))
    # dir = os.path.join(os.path.dirname(dir), "data/")  # directory of single training instances

    pretrain_dataset = MyMBKDataset('mbk_train.txt', datadir)
    dev_dataset = MyMBKDataset('mbk_dev.txt', datadir)

    # Currently batch size set to 1. Padding required for >1 batch size.
    pretrain_loader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss()
    #optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.001)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)

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
        interval = int(total / batch_size / 5)

        # scheduler.step()
        model.train()
        for (input_val, label) in pretrain_loader:
            optim.zero_grad()

            prediction, _ = model(to_variable(input_val))

            label = label.transpose_(0, 1).long().resize_(batch_size)
            # print(label)
            loss = loss_fn(prediction, to_variable(label))
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            optim.step()

            # Gradient clipping with maximum norm 0.25
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            if counter % interval == 0:
                print('Train Loss: %.2f  Progress: %d%%' % (np.asscalar(np.mean(losses)), counter * 100 / total))
            counter += 1

        print("Epoch {} Loss: {:.4f}".format(epoch, np.asscalar(np.mean(losses))))

        # validation
        count_match = 0
        losses = []
        rmse_sum = 0.0
        rmse_count = 0
        model.eval()
        for (input_val, label) in dev_loader:
            prediction, _ = model(to_variable(input_val))

            label = label.transpose_(0, 1).long().resize_(batch_size)
            loss = loss_fn(prediction, to_variable(label))
            lossnp = loss.data.cpu().numpy()
            losses.append(lossnp)
            
            prediction2 = prediction.data.cpu().numpy()
            prediction3 = np.argmax(prediction2, axis=1)
            
            #print (prediction2.shape)
            
            label_array = np.zeros((batch_size, prediction2.shape[1]))
            label_array[0][label.numpy()] = 1
            
            #print(label.numpy())
            #print(label_array[0])
            #print(prediction2[0])
            rmse = sqrt(mean_squared_error(label_array[0], prediction2[0]))
            #print(rmse)
            rmse_sum += rmse
            rmse_count += 1
                        
            if prediction3 == label.numpy():
                count_match += 1

        dev_loss = np.asscalar(np.mean(losses))
        if dev_loss < best_loss:
            torch.save(model.state_dict(), 'best_state_2')
            best_loss = dev_loss

        print("Accuracy: " + str(count_match) + " matches!")
        print("RMSE: " + str(rmse_sum/rmse_count))
        print("Epoch {} Validation Loss: {:.4f}".format(epoch, dev_loss))


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
    classes = 11
    prev_state = None
    if len(sys.argv) == 2:
        prev_state = sys.argv[1]
    main(num_of_classes=classes, datadir='./mbk', prev_state=prev_state, lr=0.0001, epochs=1000)


