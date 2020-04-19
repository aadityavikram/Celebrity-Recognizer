from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, ToTensor


def load_data(batch_size=32, source='data/celeb'):
    data = datasets.ImageFolder(root=source,
                                transform=Compose([Resize((224, 224)),
                                                   RandomHorizontalFlip(),
                                                   ToTensor()]))

    data_loader = DataLoader(data,
                             batch_size=batch_size,
                             shuffle=True)

    return data_loader
