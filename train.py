import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam

from model import DenoiseAutoEncoderModel

import argparse
from tqdm import tqdm


def add_noise(tensor, var=None, mean=0):
    if var is None:
        var = torch.FloatTensor(tensor.size()[0]).uniform_(0.4, 0.9)
        var = torch.reshape(var, (-1, 1, 1, 1))
    return tensor + torch.randn(tensor.size()) * var + mean


def train(model, train_loader, val_loader, criterion, optimizer, device, scheduler, path_to_save, epochs=10):
    best_score = None
    for epoch_idx in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (imgs, _) in tqdm(enumerate(train_loader),
                                         desc='TRAIN epoch # %d' % (epochs + 1)):
            imgs_noisy = add_noise(imgs).to(device)
            imgs = imgs.to(device)

            optimizer.zero_grad()
            outputs = model(imgs_noisy)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (imgs, _) in val_loader:
                imgs_noisy = add_noise(imgs).to(device)
                imgs = imgs.to(device)
                output = model.forward(imgs_noisy)
                loss = criterion(output, imgs)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)

        print('Epoch {} of {}, Train Loss: {:.3f}, Val loss: {:.3f}'.format(
            epoch_idx + 1, epochs, train_loss, val_loss))

        if best_score is None:
            best_score = val_loss
        elif val_loss < best_score:
            best_score = val_loss
            torch.save(model.state_dict(), path_to_save)

    print(f'\nBest model with loss {best_score} saved as {path_to_save}')


def main(args):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    batch_size = 16
    epochs = 10

    train_val_set = torchvision.datasets.MNIST(root='./data', train=True,
                                               download=True, transform=transform)

    train_set, val_set = torch.utils.data.random_split(train_val_set, [50000, 10000])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
   
    model = DenoiseAutoEncoderModel()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)

    train(model, train_loader, val_loader, criterion, optimizer, device, scheduler,
          args.save_model_path, epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_model_path', '-m',
                        help='Path to save the model', default='./net_model.pth')
    # parser.add_argument('--epoch_num', '-e', type=int, default=10,
    #                     help='Number of epochs')
    # parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    args = parser.parse_args()
    main(args)
