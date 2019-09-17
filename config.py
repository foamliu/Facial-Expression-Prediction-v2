import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
im_size = 224
num_channels = 3
num_classes = 7

# Training parameters
num_workers = 4  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none
loss_ratio = 100

train_data = 'fer2013/train'
valid_data = 'fer2013/valid'
num_train_samples = 28709
num_valid_samples = 3589
