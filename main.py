import torch
import numpy as np
from test import test
from torch import nn, optim
from load_data import load_data
from torch.autograd import Variable
from time import time, strftime, gmtime
from torchvision.models import resnet50
from utility import visualize, save_checkpoint, save_metric


def train(num_epochs=10, data_loader=None, optimizer=None, device='cuda', model=None, criterion=None, mod=50, batch_size=32):
    losses, accuracy = [], []
    print("Training....")
    start = time()
    for epoch in range(num_epochs):
        epoch_start = time()
        for i, (img, label) in enumerate(data_loader):
            optimizer.zero_grad()
            label = torch.LongTensor(label.view(-1).numpy())
            img, label = Variable(img.to(device)), Variable(label.to(device))

            # forward pass
            output = model(img)
            loss = criterion(output, label)

            # backward pass
            loss.backward()
            optimizer.step()

            pred_prob, pred_label = torch.max(output, dim=1)
            acc = (pred_label == label).sum().item() * 1.0 / batch_size

            if i % mod == 0:
                print('{} --> {}'.format(label[:4], pred_label[:4]))
                print('Epoch = {} | Batch = {} | Loss = {:.6f} | Accuracy = {:.6f}'.format(epoch + 1, i + 1, loss, acc))
                losses.append(loss)
                accuracy.append(acc)
        save_checkpoint(model=model)
        save_metric(metric=losses, name='losses_local.npy', result_dir='result')
        save_metric(metric=accuracy, name='accuracy_local.npy', result_dir='result')
        end = time()
        print('Time Elapsed in Epoch {} --> {}'.format(epoch + 1, strftime('%H:%M:%S', gmtime(end - epoch_start))))
    end = time()
    print('Training Done | Time Elapsed --> {}'.format(strftime('%H:%M:%S', gmtime(end - start))))


def main():
    torch.manual_seed(1)
    device = 'cuda'
    source = 'data/celeb'
    num_epochs = 10
    batch_size = 32
    lr = 1e-3
    mod = 50
    model = resnet50(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    data_loader = load_data(batch_size=batch_size, source=source)
    tags = np.load('data/tags.npy')

    # visualize(data_loader=data_loader, tags=tags)

    train(num_epochs=num_epochs,
          data_loader=data_loader,
          optimizer=optimizer,
          device=device,
          model=model,
          criterion=criterion,
          mod=mod,
          batch_size=batch_size)


if __name__ == '__main__':
    print('Training on {}'.format(torch.cuda.get_device_name(0)))
    main()
