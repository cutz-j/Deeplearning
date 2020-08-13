import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torchvision.transforms as transforms

# Dataset Class
class FashionMnist(Dataset):
    def __init__(self, data_path, is_train=True):
        filename = os.path.join(data_path, 'fashion-mnist_train.csv' if is_train else 'fashion-mnist_test.csv')
        assert os.path.exists(filename), 'File not found error'
        self.is_train = is_train
        self.data = pd.read_csv(filename)
        self.data = self.data.sort_values(by=['label']).to_numpy(dtype=np.float32)
        self.data_y = self.data[:, 0].astype(np.int)
        self.data_x = self.data[:, 1:]
        self.data_shape = (28, 28)
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(75),
                transforms.ToTensor()
            ]
        )

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, index):
        ret_x = np.reshape(self.data_x[index], self.data_shape)
        if self.is_train: # augmentation
            ret_x = self.transform(ret_x)
        ret_y = self.data_y[index]
        return {
            'data_x': ret_x,
            'data_y': ret_y
        }

# Model Class
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    
    def forward(self, tensor):
        return self.block(tensor)
    
class BasicModel(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(BasicModel, self).__init__()
        k = 32
        self.block = nn.Sequential(
            ConvBlock(in_channel, k),
            ConvBlock(k, k*2),
            ConvBlock(k*2, k*4),
            ConvBlock(k*4, k*2)
        )
        self.linear = nn.Linear(k*2, num_classes)
    
    def forward(self, tensor):
        out = self.block(tensor)
        out = out.view(-1, out.size(1))
        return self.linear(out)

def main():
    batch_size = 3
    epoch = 128
    learning_rate = 2e-1
    in_channel = 3
    num_classes = 10
    betas = (0.5, 0.999)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = FashionMnist('./fashion_mnist', True)
    valid_dataset = FashionMnist('./fashion_mnist', False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, pin_memory=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=False, pin_memory=True,
                              num_workers=4)

    model = BasicModel(in_channel=in_channel, num_classes=num_classes).to(device)
    criterion = nn.BCELoss() # Binary Cross Entropy Loss
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas)
    for ep in range(epoch):
        # train
        avg_loss = 0
        avg_acc = 0

        count = 0
        for idx, batch in train_loader:
            optim.zero_grad()
            batch_x = batch['data_x']
            batch_y = batch['data_y']
            output = model(batch_x)

            loss = criterion(output, batch_y)
            avg_loss += loss.item()

            # cal accuracy
            _, index = torch.max(output, 1)
            avg_acc += (index == batch_y).sum().float() / len(batch_y)
            count += 1

            loss.backward()
            optim.step()

        avg_loss /= count
        avg_acc /= count


        # valid
        avg_test_loss = 0
        avg_test_acc = 0

        count = 0
        for idx, batch in valid_loader:
            optim.zero_grad()
            batch_x = batch['data_x']
            batch_y = batch['data_y']
            output = model(batch_x)

            loss = criterion(output, batch_y)
            avg_test_loss += loss.item()

            # cal accuracy
            _, index = torch.max(output, 1)
            avg_test_acc += (index == batch_y).sum().float() / len(batch_y)
            count += 1

        avg_test_loss /= count
        avg_test_acc /= count

        print("[Epoch:%03d] train loss: %.5f train accuracy: %.4f | valid loss: %.5f valid accuracy: %.4f"
              % (ep+1, avg_loss, avg_acc, avg_test_loss, avg_test_acc))

    print("Training Done.")



if __name__ == "__main__":
    main()