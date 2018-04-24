from resnet import *
from train_resnet import get_class_num
import os
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pickle


class ClassificationDataset(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.data_files = os.listdir(dir)  # loads a list of files in __init__
        # Select only numpy files
        self.data_files = [file for file in self.data_files if file.endswith(".npy")]

        # Get total number of classes and save into a dictionary
        cnt = 0
        self.label_dict = {}
        for data_file in self.data_files:
            person = data_file.split("-")[0]
            if person not in self.label_dict:
                self.label_dict[person] = cnt
                cnt += 1
        self.total_labels = len(self.label_dict)

    def __getitem__(self, item):
        # Get training data
        filename = self.data_files[item]
        print(filename)
        X = np.load(self.dir+filename)

        # Build data label one-hot vector
        person = filename.split("-")[0]
        idx = np.array([self.label_dict[person]])
        # Y = np.zeros([self.total_labels], dtype=float)
        # Y[idx] = 1

        return filename, to_tensor(X), to_tensor(idx)

    def __len__(self):
        return len(self.data_files)


def classify(num_classes):
    batch_size = 1
    wrong_pred_file = "wrong_classification.pickle"
    cls_dir = "features_all/"
    misclassied = {}

    # Load dataset
    dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(os.path.dirname(dir), "data/")  # directory of single training instances
    classification_dataset = ClassificationDataset(dir)
    dataloader = torch.utils.data.DataLoader(classification_dataset, batch_size=batch_size, shuffle=False)

    # Load Model
    model_path = os.path.join(os.path.dirname(__file__), 'best_model')
    model = DeepSpeakerModel(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=lambda store, loc: store))

    if torch.cuda.is_available():
        # Move the network and the optimizer to the GPU
        model = model.cuda()

    for (filename, input_val, label) in dataloader:
        prediction, feats = model(to_variable(input_val))
        prediction = torch.max(prediction, dim=1)[1].cpu().data.numpy()[0]
        label = label[0].numpy()[0]

        person = filename.split("-")[0]
        np.save(feats, cls_dir+filename)    # Save feature vector for current data

        if int(prediction) != int(label):
            if person not in misclassied:
                misclassied[person] = {}

            misclassied[person][filename] = [prediction, label]

    pickle.dump(misclassied, wrong_pred_file)


if __name__ == "__main__":
    classes = get_class_num()
    classify(classes)