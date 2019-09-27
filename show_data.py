# -*- coding: utf-8 -*-

import pickle

from config import data_file

if __name__ == '__main__':
    with open(data_file, 'rb') as file:
        data = pickle.load(file)

    samples = data['train']
    data = {}
    for sample in samples:
        label = sample['label']
        if label in data:
            data[label] += 1
        else:
            data[label] = 1

    print(data)
