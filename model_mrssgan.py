import numpy as np
import torch.nn as nn
import pdb
class Discriminator(nn.Module):

    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        ##imgf-t:45-5 
        self.net = nn.Sequential(            
            nn.Conv2d(1, 32, (7,3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 48, (7,3), stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Dropout(.2),
            nn.Conv2d(48, 48, (7,3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(48, 64, (7,3), stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Dropout(.2),
            nn.Conv2d(64, 64, (7,3), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, (1,1), stride=1, padding=0),
            nn.LeakyReLU(),
            Flatten()
        )
        
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        features = self.net(x)#input:[Bs, 1, 45, 5],output: [Bs, 64]   
        logits = self.fc(features)
        return features, logits


class Generator(nn.Module):

    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        ##imgf-t:45-5
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 3),
            nn.BatchNorm1d(64 * 8 * 3),
            nn.ReLU(),
            Reshape((64, 8, 3)),
            nn.ConvTranspose2d(64, 48, 1, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 32, 1, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        return self.net(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Reshape(nn.Module):
    def __init__(self, target_shape):
        super(Reshape, self).__init__()
        self.target_shape = (-1,) + target_shape

    def forward(self, x):
        return x.view(self.target_shape)


def weights_init(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=.0, std=.1)
        nn.init.constant_(m.bias, .0)

    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, mean=0, std=.05)

