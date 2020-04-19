import os
import time
import torch
import shutil
import zipfile
import numpy as np
from torchvision.transforms import ToPILImage


def visualize(data_loader=None, tags=[]):
    images, label = iter(data_loader).next()
    idx = 1
    print('Celebrity --> {}'.format(tags[label[idx]]))
    transform = ToPILImage()
    img = transform(images[idx])
    img.show()


def save_checkpoint(model=None):
    if not os.path.exists('model'):
        os.makedirs('model')
    torch.save(model, "model/model_local.pt")


def save_metric(metric=[], name="losses.npy", result_dir="result"):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    metric_name = os.path.join(result_dir, name)
    np.save(metric_name, metric)


def extract(source='data/celeb.zip'):
    if not os.path.exists(source):
        print('Dataset zip not found or already extracted')
    else:
        print('Dataset zip found. Extracting....')
        zip_file = source
        zip_ref = zipfile.ZipFile(zip_file, 'r')
        start = time.time()
        zip_ref.extractall(path="data")
        zip_ref.close()
        print('Extracted | Time elapsed --> {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start))))


def move_data(source='data/celeb'):
    print('Moving data to source....')
    start = time.time()
    for dirs in os.listdir(source):
        dire = os.path.join(source, dirs)
        for file in os.listdir(dire):
            src = os.path.join(dire, file)
            dst = os.path.join(source, file)
            shutil.move(src, dst)
        os.rmdir(dire)
    print('Moved Data to source folder | Time Elapsed --> {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start))))


def prepare_data():
    extract(source='data/celeb.zip')
    move_data(source='data/celeb')
