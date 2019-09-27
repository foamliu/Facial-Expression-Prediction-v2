# import the necessary packages
import cv2 as cv
import numpy as np
import torch
from torchvision import transforms

from config import device, im_size
from data_gen import data_transforms
from models import FaceExpressionModel

if __name__ == '__main__':
    num_classes = 7
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    # emotion = {0:'愤怒', 1:'厌恶', 2:'恐惧', 3:'高兴', 4:'悲伤', 5:'惊讶', 6: '无表情'}

    transformer = data_transforms['valid']

    checkpoint = 'facial_expression.pt'
    print('loading model: {}...'.format(checkpoint))
    model = FaceExpressionModel()
    model.load_state_dict(torch.load(checkpoint))
    model = model.to(device)
    model.eval()

    filename = 'images/test_image_happy.jpg'
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
    print(class_names[pred])
