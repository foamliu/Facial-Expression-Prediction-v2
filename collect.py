import pickle

import cv2 as cv

if __name__ == "__main__":
    with open('fer2013.pkl', 'rb') as file:
        data = pickle.load(file)

    train = data['train']
    train = train[:10]

    for i, sample in enumerate(train):
        filename = sample['image_path']
        img = cv.imread(filename)
        new_name = 'images/{}.jpg'.format(i)
        cv.imwrite(new_name, img)
        label = sample['label']
        print(label)
