import math
import os

IMAGE_WIDTH = 480
IMAGE_HEIGHT = 360


def normalize(x, y, w, h):
    x_norm = (x + w / 2) / IMAGE_WIDTH
    y_norm = (y + h / 2) / IMAGE_HEIGHT
    w_norm = w / IMAGE_WIDTH
    h_norm = h / IMAGE_HEIGHT
    return x_norm, y_norm, w_norm, h_norm


def denormalize(x_norm, y_norm, w_norm, h_norm):
    w = round(w_norm * IMAGE_WIDTH)
    h = round(h_norm * IMAGE_HEIGHT)
    x = round(x_norm * IMAGE_WIDTH - w / 2)
    y = round(y_norm * IMAGE_HEIGHT - h / 2)
    return [x, y, w, h]


def normalize_annotations(dirname):
    for filename in os.listdir(dirname):
        if filename.endswith('.txt'):
            with open(f'./data/test/{filename}', 'r') as f:
                label, x, y, w, h, _ = f.readline().split(' ')
                x, y, w, h = normalize(int(x), int(y), int(w), int(h))
                with open(f'./data/test/{filename}', 'w') as wf:
                    print(f'{label} {x} {y} {w} {h}', file=wf)


if __name__ == '__main__':
    normalize_annotations('./data/test')
    normalize_annotations('./data/train')
