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
    y_pred = []

    start = time.time()

    for sample in tqdm(samples):
        filename = sample['image_path']
        img = cv.imread(filename)
        img = cv.resize(img, (im_size, im_size))
        img = img[..., ::-1]
        img = transforms.ToPILImage()(img)
        img = transformer(img)
        img = torch.unsqueeze(img, dim=0)
        img = img.to(device)
        with torch.no_grad():
            pred = model(img)[0]
        pred = pred.cpu().numpy()
        pred = np.argmax(pred)
        y_pred.append(pred)

    end = time.time()
    seconds = end - start
    print('avg fps: {}'.format(str(len(samples) / seconds)))

    return y_pred, y_test


class FaceExpressionEmbedder(nn.Module):
    def __init__(self):
        super(FaceExpressionEmbedder, self).__init__()
        checkpoint = 'facial_expression.pt'
        print('loading model: {}...'.format(checkpoint))
        model = FaceExpressionModel()
        model.load_state_dict(torch.load(checkpoint))
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)

    def forward(self, images):
        x = self.model(images)  # [N, 2048, 1, 1]
        return x


if __name__ == '__main__':
    embedder = FaceExpressionEmbedder()
    model = FaceExpressionModel().to(device)
    summary(model, input_size=(3, 112, 112))

    # model = embedder.to(device)
    # model.eval()
    #
    # num_classes = 7
    # class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    # # emotion = {0:'愤怒', 1:'厌恶', 2:'恐惧', 3:'高兴', 4:'悲伤', 5:'惊讶', 6: '无表情'}
    #
    # with open(data_file, 'rb') as file:
    #     data = pickle.load(file)
    #
    # samples = data['test']
    # transformer = data_transforms['valid']
    #
    # y_pred, y_test = predict(model, samples)
