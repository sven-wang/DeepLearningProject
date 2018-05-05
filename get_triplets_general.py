from resnet_2d import *
from train_resnet_test_small import get_class_num
import os
import torch
from torch.autograd import Variable
import numpy as np
import pickle
import pdb
from scipy.spatial.distance import cosine
import random


def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return Variable(tensor)


class ClassificationDataset(Dataset):
    def __init__(self, dir, train_file, dev_file, person2label_map):
        self.dir = dir
        self.data_files = []
        with open(train_file) as f:
            self.data_files.extend(f.readlines())
        with open(dev_file) as f:
            self.data_files.extend(f.readlines())

        with open(person2label_map, 'rb') as handle:
            self.label_dict = pickle.load(handle)

        self.total_labels = len(self.label_dict)

    def __getitem__(self, item):
        # Get training data
        filename = self.data_files[item].strip()
        X = np.load(self.dir + filename)

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
    cls_dir = "./vectors/"
    person2label_map = "person2label_map.pickle"
    train_file = "train2.txt"
    dev_file = "dev2.txt"
    misclassied = {}

    # Load dataset
    dir = "./new_features/"   # directory of single training instances
    classification_dataset = ClassificationDataset(dir, train_file, dev_file, person2label_map)
    dataloader = torch.utils.data.DataLoader(classification_dataset, batch_size=batch_size, shuffle=False)

    # Load Model
    model_path = "./best_state"
    model = DeepSpeakerModel(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.eval()

    if torch.cuda.is_available():
        # Move the network and the optimizer to the GPU
        model = model.cuda()
    for (filename, input_val, label) in dataloader:
        filename = filename[0]
        prediction, feats = model(to_variable(input_val))

        prediction = torch.max(prediction, dim=1)[1].cpu().data.numpy()[0]
        label = label[0].numpy()[0]

        person = filename.split("-")[0]

        np.save(cls_dir + filename[:-4], feats.data.cpu().numpy())    # Save feature vector for current data

        if int(prediction) != int(label):
            if person not in misclassied:
                misclassied[person] = {}

            misclassied[person][filename] = (prediction, int(label))

    with open(wrong_pred_file, 'wb') as handle:
        pickle.dump(misclassied, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_misclassified_triplets(misclassified):
    triplets = []
    label2person_map = "label2person_map.pickle"
    with open(label2person_map, 'rb') as handle:
        label2person = pickle.load(handle)

    # Build a dictionary storing each person's files
    vector_dir = "./vectors/"
    file_dict = {}
    files = os.listdir(vector_dir)
    added_triplets = set([])
    for filename in files:
        person = filename.split("-")[0]
        if person not in file_dict:
            file_dict[person] = []
        file_dict[person].append(filename)

    # Iterate all wrongly classified files
    for person in misclassified:
        for anchor_file in misclassified[person]:
            anchor = anchor_file.split("-")[0]
            anchor_vec = np.load(vector_dir + anchor_file)

            negative_person = label2person[misclassified[person][anchor_file][0]]
            d_a_n = {}
            for negative_file in file_dict[negative_person]:
                # Distance between Anchor and Negative
                d_a_n[negative_file] = cosine(anchor_vec, np.load(vector_dir + negative_file))

            # Iterate all positive files for the same person
            for positive_file in file_dict[anchor]:
                if positive_file == anchor_file:
                    continue

                # Cosine distance (1 - similarity) between Anchor and Positive
                d_a_p = cosine(anchor_vec, np.load(vector_dir + positive_file))

                for negative_file in d_a_n:
                    # Compare Distance. If condition satisfied, add the triplet.
                    if d_a_n[negative_file] < d_a_p:
                        triplets.append((anchor_file, positive_file, negative_file))
                        temp = anchor_file + "#" + positive_file + "#" + negative_file
                        added_triplets.add(temp)

    return triplets, added_triplets


def get_general_triplets(added_triplets):
    all_file_path = "all.txt"
    triplets = []
    persons = {}

    with open(all_file_path, "r") as f:
        all_files = f.readlines()
        for each_file in all_files:
            each_file = each_file.strip().split()[-1]
            person = each_file.split("-")[0]
            if person not in persons:
                persons[person] = []
            persons[person].append(each_file)

    for person in persons:
        # Choose current person as the anchor person
        files = persons[person]
        # Skip if this person only has one file
        if len(files) < 2:
            continue

        for i in range(len(files) - 1):
            anchor_file = files[i]
            # Randomly choose a positive file
            positive_file = files[random.randint(i + 1, len(files) - 1)]

            # For rest of other persons, choose one for each as negative files
            for diff_person in persons:
                if diff_person == person:
                    continue

                diff_files = persons[diff_person]
                negative_file = diff_files[random.randint(0, len(diff_files) - 1)]

                temp = anchor_file + "#" + positive_file + "#" + negative_file
                if temp in added_triplets:
                    continue
                added_triplets.add(temp)

                triplets.append((anchor_file, positive_file, negative_file))

    return triplets


if __name__ == "__main__":
    # classes = get_class_num()
    classes = 834
    classify(classes)
    print("Finished Classifying!")

    with open("wrong_classification.pickle", "rb") as handle:
        misclassified = pickle.load(handle)

    triplets, added_triplets = get_misclassified_triplets(misclassified)
    print("Finished generating misclassified triplets!")

    # Write all triplets to a file
    output_file = open("triplets_misclassified.csv", 'w')
    for triplet in triplets:
        output_file.write(triplet[0] + "," + triplet[1] + "," + triplet[2] + "\n")
    output_file.close()

    # Randomly sample triplets
    triplets = random.choices(triplets, k=int(0.6 * len(triplets)))

    general_triplets = get_general_triplets(added_triplets)

    # Write all triplets to a file
    output_file = open("triplets_general.csv", 'w')
    for triplet in general_triplets:
        output_file.write(triplet[0] + "," + triplet[1] + "," + triplet[2] + "\n")
    output_file.close()

    triplets.extend(general_triplets)
    # Write all triplets to a file
    output_file = open("triplets.csv", 'w')
    for triplet in triplets:
        output_file.write(triplet[0] + "," + triplet[1] + "," + triplet[2] + "\n")
    output_file.close()
    print("All done!")
