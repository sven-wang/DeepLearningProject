'''
Sampling logic:
If no more than 4 audios, all go to train.  9542
Else, sample [0.9*num] for train, the rest go to dev 1405
'''

import collections
import random

all_file = open('all.txt')
train_file = open('train.txt', 'w')
dev_file = open('dev.txt', 'w')



def write_list(file, towrite):
    for each in towrite:
        file.write(each + '\n')


speaker_dict = collections.defaultdict(list)

for line in all_file:
    filename = line.strip().split()[-1]
    speaker = filename.split('-')[0]
    speaker_dict[speaker].append(filename)

print(len(speaker_dict))  # 1319

for speaker in speaker_dict:
    speeches = speaker_dict[speaker]

    if len(speeches) <= 4:
        write_list(train_file, speeches)
    else:
        random.shuffle(speeches)
        write_list(train_file, speeches[:int(len(speeches) * 0.9)])
        write_list(dev_file, speeches[int(len(speeches) * 0.9):])







