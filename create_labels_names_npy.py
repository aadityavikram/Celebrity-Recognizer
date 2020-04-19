import numpy as np
from os import path, listdir

root = 'data/celeb'
labels, names = [], []
for i, dirs in enumerate(listdir(root)):
    for images in listdir(path.join(root, dirs)):
        labels.append([i + 1])
        names.append(images)

np.save('data/labels.npy', labels)
np.save('data/names.npy', names)
