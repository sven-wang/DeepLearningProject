from resnet_2d_small import *
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


def train():
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(999)

    train_dataset = TripletDataset("triplets_sample.csv", "new_features/")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load Model
    model_path = os.path.join(os.path.dirname(__file__), 'best_state_small')
    model = DeepSpeakerModel(classes)
    model.load_state_dict(torch.load(model_path))

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)

    losses = []
    total = len(train_dataset)
    for epoch in range(epochs):
        # for EER
        y_gold = []
        y_pred = []

        counter = 1
        model.train()
        for (data_a, data_p, data_n) in train_dataloader:
            data_a, data_p, data_n = to_variable(data_a), to_variable(data_p), to_variable(data_n)

            # compute output
            out_a, out_p, out_n = model(data_a)[1], model(data_p)[1], model(data_n)[1]  # vector before the fc layer

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

            if counter % 3000 == 0:
                print('Train Loss: %.2f  Progress: %d%%' % (np.asscalar(np.mean(losses)), counter * 100 / total))
                print('EER:', eer(y_gold, y_pred))
            counter += 1

        print("Epoch {} Loss: {:.4f}".format(epoch, np.asscalar(np.mean(losses))))
        print('EER:', eer(y_gold, y_pred))

        torch.save(model.state_dict(), 'best_state_triplet_small')


if __name__ == "__main__":
    batch_size = 8
    lr = 0.001
    epochs = 100
    classes = 1303
    margin = 0.1  # the margin value for the triplet loss function (default: 1.0)

    train()
