import torch
from warpctc_pytorch import CTCLoss
from ctcdecode import CTCBeamDecoder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.nn.functional import softmax
import os
import numpy as np
from phoneme_list import PHONEME_MAP


def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()


def to_int_tensor(numpy_array):
    return torch.from_numpy(numpy_array).int()


class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, item):
        return to_tensor(self.X[item]), to_tensor(self.Y[item])

    def __len__(self):
        return self.X.shape[0]


def collate(data):
    data.sort(key=lambda x: x[0].shape[0], reverse=True)
    max_len = data[0][0].shape[0]
    seq_len = []
    label_len = []
    batchX = []

    utts_tup, phonemes_tup = zip(*data)
    for i in range(len(utts_tup)):
        seq_len.append(utts_tup[i].shape[0])
        label_len.append(phonemes_tup[i].shape[0])

        itemX = np.pad(utts_tup[i], [(0, max_len - utts_tup[i].shape[0]), (0, 0)], mode="constant")
        batchX.append(itemX)
    concat_labels = np.concatenate(phonemes_tup)

    return to_tensor(np.array(batchX)), \
           to_int_tensor(concat_labels), \
           np.array(seq_len), \
           to_int_tensor(np.array(label_len))


def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)

# read test data
sample = np.load("sample_data_mtx.np.npy")
testY = np.zeros([sample.shape[0], sample.shape[1]])
test_dataset = CustomDataset(sample, testY)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate)

model_path = os.path.join(os.path.dirname(__file__), 'model_1333.222900390625')
model = LSTMmodel2(256)
model.load_state_dict(torch.load(model_path, map_location=lambda store, loc: store))
model = model.cuda()
model.eval()
res = []

label_map = [' '] + PHONEME_MAP
decoder = CTCBeamDecoder(
    labels=label_map,
    blank_id=0
)
for (input, targ, seq_len, label_len) in test_loader:
    logits = model(to_variable(input), seq_len)
    logits = torch.transpose(logits, 0, 1)
    probs = softmax(logits, dim=2).data.cpu()

    output, scores, timesteps, out_seq_len = \
        decoder.decode(probs=probs, seq_lens=to_int_tensor(seq_len))
    for i in range(output.size(0)):
        chrs = "".join(label_map[o] for o in output[i, 0, :out_seq_len[i, 0]])
        res.append(chrs)

print(res)
