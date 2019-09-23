import math
import pickle

import cv2 as cv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from align_faces import get_reference_facial_points, warp_and_crop_face
from config import device, im_size
from mtcnn.detector import detect_faces

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def align_face(img_fn, facial5points):
    raw = cv.imread(img_fn, True)
    facial5points = np.reshape(facial5points, (2, 5))

    crop_size = (im_size, im_size)

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    output_size = (im_size, im_size)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    # dst_img = warp_and_crop_face(raw, facial5points)
    dst_img = warp_and_crop_face(raw, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
    return dst_img


def select_central_face(im_size, bounding_boxes):
    width, height = im_size
    nearest_index = -1
    nearest_distance = 100000
    for i, b in enumerate(bounding_boxes):
        x_box_center = (b[0] + b[2]) / 2
        y_box_center = (b[1] + b[3]) / 2
        x_img = width / 2
        y_img = height / 2
        distance = math.sqrt((x_box_center - x_img) ** 2 + (y_box_center - y_img) ** 2)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_index = i

    return nearest_index


def get_central_face_attributes(full_path):
    img = Image.open(full_path).convert('RGB')
    bounding_boxes, landmarks = detect_faces(img)

    if len(landmarks) > 0:
        i = select_central_face(img.size, bounding_boxes)
        return True, [bounding_boxes[i]], [landmarks[i]]


def get_all_face_attributes(full_path):
    img = Image.open(full_path).convert('RGB')
    bounding_boxes, landmarks = detect_faces(img)
    return bounding_boxes, landmarks


class FaceExpressionDataset(Dataset):
    def __init__(self, split):
        with open('fer2013.pkl', 'rb') as file:
            data = pickle.load(file)

        self.samples = data[split]
        self.transformer = data_transforms[split]

    def get_image(self, filename):
        img = cv.imread(filename)
        img = img[..., ::-1]
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)
        img = img.to(device)
        return img

    def __getitem__(self, i):
        sample = self.samples[i]
        full_path = sample['image_path']
        img = self.get_image(full_path)
        print(img.size())

        label = sample['label']
        return img, label

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    dataset = FaceExpressionDataset('train')
    print(dataset[0])
