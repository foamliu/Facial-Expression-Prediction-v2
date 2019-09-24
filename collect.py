import pickle

import cv2 as cv


class FaceExpressionDataset(Dataset):
    def __init__(self, split):
        self.samples = data[split]
        self.transformer = data_transforms[split]

    def __getitem__(self, i):
        sample = self.samples[i]
        filename = sample['image_path']
        img = cv.imread(filename)
        img = cv.resize(img, (im_size, im_size))
        img = img[..., ::-1]
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)
        label = sample['label']
        return img, label

    def __len__(self):
        return len(self.samples)


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
