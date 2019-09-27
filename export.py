import time

import torch

from models import FaceExpressionModel

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    print('loading {}...'.format(checkpoint))
    start = time.time()
    checkpoint = torch.load(checkpoint)
    print('elapsed {} sec'.format(time.time() - start))
    model = checkpoint['model']
    print(model)
    print(type(model))

    # model.eval()
    filename = 'facial_expression.pt'
    print('saving {}...'.format(filename))
    start = time.time()
    torch.save(model.state_dict(), filename)
    print('elapsed {} sec'.format(time.time() - start))

    print('loading {}...'.format(filename))
    start = time.time()
    model = FaceExpressionModel()
    model.load_state_dict(torch.load(filename))
    print('elapsed {} sec'.format(time.time() - start))
