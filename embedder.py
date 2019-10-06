# -*- coding: utf-8 -*-

import pickle
import time

import matplotlib as mpl
from matplotlib import pylab

mpl.use('tkagg')
from sklearn.manifold import TSNE

import cv2 as cv
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from config import device, im_size, data_file, num_classes
from data_gen import data_transforms
from models import FaceExpressionModel


def predict(model, samples):
    embeddings = np.zeros((num_samples, 2048), dtype=np.float)
    labels = []

    start = time.time()

    for i in tqdm(range(num_samples)):
        sample = samples[i]
        filename = sample['image_path']
        img = cv.imread(filename)
        img = cv.resize(img, (im_size, im_size))
        img = img[..., ::-1]
        img = transforms.ToPILImage()(img)
        img = transformer(img)
        img = torch.unsqueeze(img, dim=0)
        img = img.to(device)

        with torch.no_grad():
            embedded = model(img)[0]
            embedded = embedded.cpu().numpy()

        # print(embedded.shape)
        embeddings[i] = embedded

        labels.append(sample['label'])

    end = time.time()
    elapsed = end - start
    print('seconds per image: {:.4f}'.format(elapsed / num_samples))

    return embeddings, labels


class FaceExpressionEmbedder(nn.Module):
    def __init__(self):
        super(FaceExpressionEmbedder, self).__init__()
        checkpoint = 'facial_expression.pt'
        print('loading model: {}...'.format(checkpoint))
        model = FaceExpressionModel()
        model.load_state_dict(torch.load(checkpoint))
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

    def forward(self, images):
        x = self.model(images)  # [N, 2048, 1, 1]
        x = x.view(-1, 2048)  # [N, 2048]
        return x


def get_cmap():
    cmap = pylab.cm.jet  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (.5, .5, .5, 1.0)

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(0, num_classes, num_classes + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def get_annotations(two_d, labels):
    print('two_d.shape: ' + str(two_d.shape))
    coods = {}
    for i in range(num_samples):
        x, y = two_d[i][0], two_d[i][1]
        label = labels[i]
        if label in coods:
            coods[label]['x'].append(x)
            coods[label]['y'].append(y)
        else:
            coods[label] = {'x': [], 'y': []}

    xs = []
    ys = []
    labels = []
    for label in coods.keys():
        xs.append(np.median(coods[label]['x']))
        ys.append(np.median(coods[label]['y']))
        labels.append(label)

    print('xs: ' + str(xs))
    print('ys: ' + str(ys))
    print('ids: ' + str(labels))

    return xs, ys, labels


if __name__ == '__main__':
    embedder = FaceExpressionEmbedder().to(device)
    # summary(embedder, input_size=(3, 112, 112))
    embedder = embedder.to(device)
    embedder.eval()
    #
    # num_classes = 7
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    # emotion = {0:'愤怒', 1:'厌恶', 2:'恐惧', 3:'高兴', 4:'悲伤', 5:'惊讶', 6: '无表情'}
    #
    with open(data_file, 'rb') as file:
        data = pickle.load(file)

    samples = data['test']
    num_samples = len(samples)
    print('num_samples: ' + str(num_samples))

    transformer = data_transforms['valid']

    embeddings, labels = predict(embedder, samples)
    # print(labels)

    labels = [class_names[idx] for idx in labels]
    # print(labels)

    print('t-SNE: fitting transform...')
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_embeddings = tsne.fit_transform(embeddings)

    cmap, norm = get_cmap()

    xs, ys, labels = get_annotations(two_d_embeddings, labels)

    pylab.figure(figsize=(15, 15))
    pylab.scatter(two_d_embeddings[:, 0], two_d_embeddings[:, 1], c=labels, cmap=cmap, norm=norm, alpha=0.8,
                  edgecolors='none', s=10)
    for i, label in enumerate(labels):
        pylab.annotate(label, xy=(xs[i], ys[i]), xytext=(0, 0), textcoords='offset points',
                       ha='center', va='center')
    pylab.show()
