from resnet import *
import os
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pickle
from train_resnet import get_class_num


def get_triplets(misclassified):
    dist_fn = CosineSimilarity()
    triplets = []

    # Build a dictionary storing each person's files
    feats_dir = "features_all/"
    file_dict = {}
    files = os.listdir(feats_dir)
    for filename in files:
        person = filename.split("-")[0]
        if person not in file_dict:
            file_dict[person] = []
        file_dict[person].append(filename)

    # Iterate all wrongly classified files
    for person in misclassified:
        for anchor_file in misclassified[person]:
            anchor = anchor_file.split("-")[0]
            anchor_vec = np.load(feats_dir+anchor_file)

            # Iterate all positive files for the same person
            for positive_file in file_dict[anchor]:
                if positive_file == anchor_file:
                    continue

                # Distance between Anchor and Positive
                d_a_p = dist_fn.forward(anchor_vec, np.load(feats_dir+positive_file))[0]

                # Iterate all files for other persons
                for negative_person in file_dict:
                    if negative_person == person:
                        continue
                    for negative_file in file_dict[negative_person]:

                        # Distance between Anchor and Negative
                        d_a_n = dist_fn.forward(anchor_vec, np.load(feats_dir+negative_file))[0]

                        # Compare distances. If condition satisfied, add the triplet.
                        if d_a_n > d_a_p:
                            triplets.append((anchor_file, positive_file, negative_file))

    # Write all triplets to a file
    output_file = open("triplets.csv", 'w')
    for triplet in triplets:
        output_file.write(triplet[0] + "," + triplet[1] + "," + triplet[2] + "\n")
    output_file.close()


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
