# import the necessary packages
import time

import numpy as np
import torch
from torchvision import transforms

from config import device
from data_gen import data_transforms
from data_gen import get_central_face_attributes, align_face
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

    test_images = ['images/test_image_happy.jpg', 'images/test_image_angry.jpg']
    start = time.time()
    for filename in test_images:
        has_face, bboxes, landmarks = get_central_face_attributes(filename)
        img = align_face(filename, landmarks)
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
    end = time.time()
    elapsed = end - start
    print('{} seconds per image'.format(elapsed / len(test_images)))
