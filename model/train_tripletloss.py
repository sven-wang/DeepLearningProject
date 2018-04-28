from resnet_2d_small import *
import os
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pickle


def train():
    # Construct all triplets and write to file
    with open("wrong_classification.pickle", 'rb') as handle:
        misclassified = pickle.load(handle)

    get_triplets(misclassified)

    dataset = TripletDataset("triplets.csv", "features_all/")
    # TODO: 1. Init data loader. 2. Start trainning.

    # Load Model
    model_path = os.path.join(os.path.dirname(__file__), 'beststate')
    model = DeepSpeakerModel(classes)
    model.load_state_dict(torch.load(model_path, map_location=lambda store, loc: store))

    return


if __name__ == "__main__":
    batch_size = 1
    lr = 0.001
    epochs = 20
    #classes = get_class_num()
    classes = 351
    train()
