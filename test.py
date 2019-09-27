# -*- coding: utf-8 -*-

import itertools
import pickle
import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from tqdm import tqdm

from config import device, im_size, data_file
from data_gen import data_transforms
from models import FaceExpressionModel


def read_data(file_path):
    data = pd.read_csv(file_path)
    print(data.shape)
    print(data.head())
    print(np.unique(data["Usage"].values.ravel()))
    print('The number of PrivateTest data set is %d' % (len(data[data.Usage == "PrivateTest"])))
    test_data = data[data.Usage == "PrivateTest"]
    # test_images = parse_images(test_data)
    y_test = test_data["emotion"].values.ravel()
    return y_test


def predict(model, samples):
    y_pred = []

    start = time.time()

    with torch.no_grad():
        for sample in tqdm(samples):
            filename = sample['image_path']
            img = cv.imread(filename)
            img = cv.resize(img, (im_size, im_size))
            img = img[..., ::-1]
            img = transforms.ToPILImage()(img)
            img = transformer(img)
            img = torch.unsqueeze(img, dim=0)
            img = img.to(device)
            pred = model(img)[0]
            pred = pred.cpu().numpy()
            pred = np.argmax(pred)
            y_pred.append(pred)

    end = time.time()
    seconds = end - start
    print('avg fps: {}'.format(str(num_test_samples / seconds)))

    return y_pred, y_prob


def decode(y_test):
    ret = []
    for i in range(len(y_test)):
        id = y_test[i]
        ret.append(class_names[id])
    return ret


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def calc_acc(y_pred, y_test):
    num_corrects = 0
    for i in range(num_test_samples):
        pred = y_pred[i]
        test = y_test[i]
        if pred == test:
            num_corrects += 1
    return num_corrects / num_test_samples


if __name__ == '__main__':
    img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 7
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    # emotion = {0:'愤怒', 1:'厌恶', 2:'恐惧', 3:'高兴', 4:'悲伤', 5:'惊讶', 6: '无表情'}
    num_test_samples = 3589

    checkpoint = 'facial_expression.pt'
    print('loading model: {}...'.format(checkpoint))
    model = FaceExpressionModel()
    model.load_state_dict(torch.load(checkpoint))
    model = model.to(device)
    model.eval()

    with open(data_file, 'rb') as file:
        data = pickle.load(file)

    samples = data['test']
    transformer = data_transforms['valid']

    y_pred, y_prob = predict(model, samples)
    print("y_pred: " + str(y_pred))

    y_test = read_data('fer2013/fer2013.csv')
    y_test = decode(y_test)
    print("y_test: " + str(y_test))

    acc = calc_acc(y_pred, y_test)
    print("%s: %.2f%%" % ('acc', acc * 100))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
