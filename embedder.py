# -*- coding: utf-8 -*-

import pickle
import time

import cv2 as cv
import numpy as np
import torch
from torch import nn
from torchsummary import summary
from torchvision import transforms
from tqdm import tqdm

from config import device, im_size, data_file
from config import num_classes
from data_gen import data_transforms
from models import FaceExpressionModel


def predict(model, samples):
    num_samples = len(samples)
    print('num_samples: ' + str(num_samples))
    embeddings = np.zeros((num_samples, 2048), dtype=np.float)

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

        print(embedded.shape)
        embeddings[i] = embedded
        break

    end = time.time()
    seconds = end - start
    print('avg fps: {}'.format(str(len(samples) / seconds)))

    return embeddings


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


if __name__ == '__main__':
    embedder = FaceExpressionEmbedder().to(device)
    # summary(embedder, input_size=(3, 112, 112))
    embedder = embedder.to(device)
    embedder.eval()
    #
    # num_classes = 7
    # class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    # # emotion = {0:'愤怒', 1:'厌恶', 2:'恐惧', 3:'高兴', 4:'悲伤', 5:'惊讶', 6: '无表情'}
    #
    with open(data_file, 'rb') as file:
        data = pickle.load(file)

    samples = data['test']
    transformer = data_transforms['valid']

    embedding_list = predict(embedder, samples)
