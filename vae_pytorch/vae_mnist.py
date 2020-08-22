# -*- coding: utf-8 -*-


#参考にしたサイト
#https://tips-memo.com/vae-pytorch#1


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import torch.utils as utils
from torchvision.utils import save_image
from torchvision import datasets, transforms

from sklearn.manifold import TSNE
from random import random

#############################################################################
#############################################################################
## MODEL VAE ################################################################

# 入力 & 潜在空間の次元
input_dim  = 784
layer1_dim = 400
layer2_dim = 100
latent_dim = 2

class Encoder(nn.Module):

    # note: output mu & lv(logaritm of variance)
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, layer1_dim)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(layer1_dim, layer2_dim)
        self.r2 = nn.ReLU()
        self.fc3_mu = nn.Linear(layer2_dim, latent_dim)
        self.fc3_lv = nn.Linear(layer2_dim, latent_dim)

    # 推論利用時のencoder
    def encode(self, x):
        h = self.r1(self.fc1(x))
        h = self.r2(self.fc2(h))
        return self.fc3_mu(h), self.fc3_lv(h)

    # reparameterization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    # 学習利用時のencoder
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc4 = nn.Linear(latent_dim, layer2_dim)
        self.r4  = nn.ReLU()
        self.fc5 = nn.Linear(layer2_dim, layer1_dim)
        self.r5  = nn.ReLU()
        self.fc6 = nn.Linear(layer1_dim, input_dim)
        self.s   = nn.Sigmoid()

    def decode(self, z):
        h = self.r4(self.fc4(z))
        h = self.r5(self.fc5(h))
        return self.s(self.fc6(h))

    def forward(self, z):
        return self.decode(z)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        return self.decoder(z), mu, logvar



def loss_function(rec_x, x, mu, logvar):
    # BCEはデータにより変更
    BCE = F.binary_cross_entropy(rec_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #print("loss", BCE.item(), KLD.item())
    return BCE + KLD


## MODEL VAE ################################################################
#############################################################################
#############################################################################

def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x


def Train(epoch, dataloader, model, optimizor, device):
    train_loss = 0
    for batch_idx, data_label in enumerate(dataloader):
        data, label = data_label
        #print(type(data), data.numpy().shape)
        #print(data.numpy()[10,0])
        data = data.view(data.size(0), -1)

        data = data.to(device)
        optimizor.zero_grad()
        rec_batch, mu, logvar = model(data)
        loss = loss_function(rec_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizor.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(dataloader.dataset)))
    if epoch % 10 == 0 :
        pic = to_img(rec_batch.cpu().data)
        save_image(pic, './image_{}.png'.format(epoch))



def Test(dataloader, model, device):
    total_loss = 0.0
    with torch.no_grad():
        for data_label in dataloader:
            data, label = data_label
            data = data.view(data.size(0), -1)
            data = data.to(device)
            rec, mu, logvar = model(data)
            loss = loss_function(rec, data, mu, logvar)
            total_loss += loss.item()
    return total_loss


def main():

    #load datasets
    #transform = transforms.Compose( [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    transform = transforms.Compose([transforms.ToTensor()])#もとから[0,1]なのでそれを利用
    dataset_train = datasets.MNIST( './data', train=True, download=True, transform=transform)
    dataset_test = datasets.MNIST( './data', train=False, download=True,transform=transform)
    dataloader_train = utils.data.DataLoader(dataset_train, batch_size=100, shuffle=True, num_workers=4)
    dataloader_test  = utils.data.DataLoader(dataset_test, batch_size=100, shuffle=True, num_workers=4)
    print(len(dataloader_train), type(dataloader_train))
    print(len(dataloader_test), type(dataloader_test))


    #check cuda
    if torch.cuda.is_available() :
        device = torch.device("cuda")
        print ( "cuda is available ")
    else :
        device = torch.device("cpu")
        print ( "cuda is NOT available ")

    # モデルとoptimizerの定義
    model = VAE().to(device)
    optimizor = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    # 学習エポック数
    epochs = 200
    log_loss = []
    for epoch in range(1, epochs + 1):
        Train(epoch, dataloader_train, model, optimizor, device)
        if epoch % 10 == 0:
            log_loss.append(Test(dataloader_test, model, device))
            torch.save(model.state_dict(), 'model_' + str(epoch) + '.pth')

    #出力
    torch.save(model.state_dict(), 'model_fin.pth')


    # Lossのグラフ
    x_axis = [(i+1)*10 for i in range(int(epochs/10))]
    plt.figure()
    plt.plot(x_axis, log_loss)
    plt.grid()
    plt.savefig('loss_.png')


    #洗剤空間の分布を可視化(see https://tips-memo.com/vae-pytorch#1)
    colors = ["red", "green", "blue", "orange", "purple", "brown", "fuchsia", "grey", "olive", "lightblue"]

    plt.figure(figsize=(10,10))


    model.eval()
    zs = []
    count  = 0
    for x, t in dataloader_test:
        x = x.view(x.size(0), -1).to(device)
        t = t.to(device)
        # generate from x
        y, mu, logvar = model(x)

        zs = mu.cpu().detach().numpy()
        labels = t.cpu().detach().numpy()
        points = TSNE(n_components=2, random_state=0).fit_transform(zs)
        for p, l in zip(points, labels):
          plt.scatter(p[0], p[1], marker="${}$".format(l), c=colors[l])
        count = count + 1
        if count > 5 : break
    plt.savefig('latent_distrib.png')
    plt.show()


if __name__ == '__main__':
    main()
