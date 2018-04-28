from resnet_2d_small import *
import os
import torch


def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


def train():
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(999)

    train_dataset = TripletDataset("triplets.csv", "new_features/")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load Model
    model_path = os.path.join(os.path.dirname(__file__), 'best_state_small')
    model = DeepSpeakerModel(classes)
    model.load_state_dict(torch.load(model_path))

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)

    for (data_a, data_p, data_n) in train_dataloader:

        data_a, data_p, data_n = to_variable(data_a), to_variable(data_p), to_variable(data_n)

        # compute output
        out_a, out_p, out_n = model(data_a)[0], model(data_p)[0], model(data_n)[0]  # vector before the fc layer

        triplet_loss = TripletMarginLoss(margin).forward(out_a, out_p, out_n)

        # compute gradient and update weights
        optimizer.zero_grad()
        triplet_loss.backward()
        optimizer.step()

        # print loss
        print(triplet_loss.data.cpu().numpy())




    return


if __name__ == "__main__":
    batch_size = 1
    lr = 0.001
    epochs = 100
    classes = 1303
    margin = 0.1  # the margin value for the triplet loss function (default: 1.0)

    train()
