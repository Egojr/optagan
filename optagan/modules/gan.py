import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, block_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(block_dim, block_dim),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.net(x) + x

class Generator(nn.Module):
    def __init__(self, n_layers, block_dim, latent_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, block_dim),
            nn.LeakyReLU(0.2,True),
            *[Block(block_dim) for _ in range(n_layers)],
            nn.Linear(block_dim, latent_size)
            )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, n_layers, block_dim, latent_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, block_dim),
            nn.LeakyReLU(0.2, True),
            *[Block(block_dim) for _ in range(n_layers)],
            nn.Linear(block_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x)

class cond_Generator(nn.Module):
    def __init__(self, n_layers, block_dim, latent_size, classes):
        super().__init__()
        self.label_embedding = nn.Embedding(classes, latent_size).requires_grad_(True)
        self.net = nn.Sequential(
            nn.Linear(latent_size*2, block_dim),
            nn.LeakyReLU(0.2, True),
            *[Block(block_dim) for _ in range(n_layers)],
            nn.Linear(block_dim, latent_size)
            )

    def forward(self, x, labels):
        c = self.label_embedding(labels).squeeze(1)
        x = torch.cat((x,c), 1)
        return self.net(x)

class cond_Critic(nn.Module):
    def __init__(self, n_layers, block_dim, latent_size, classes):
        super().__init__()
        self.label_embedding = nn.Embedding(classes, latent_size).requires_grad_(True)
        self.net = nn.Sequential(
            nn.Linear(latent_size*2, block_dim),
            nn.LeakyReLU(0.2, True),
            *[Block(block_dim) for _ in range(n_layers)],
            nn.Linear(block_dim, 1)
            )

    def forward(self, x, labels): 
        c = self.label_embedding(labels).squeeze(1)
        x = torch.cat((x,c), 1)
        return self.net(x)

class Classifier(nn.Module):
    def __init__(self, latent_size, block_dim, classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, block_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(block_dim, block_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(block_dim, block_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(block_dim, classes if classes > 2 else 1)
            )
            
    def forward(self, x):
        return self.net(x)

# For sliced losses
class s_Critic(nn.Module):
    def __init__(self, n_layers, block_dim, latent_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, block_dim),
            nn.LeakyReLU(0.2, True),
            *[Block(block_dim) for _ in range(n_layers)]
        )
        self.end = nn.Linear(block_dim, 1)
        
    def forward(self, x):
        out = self.net(x)
        return self.end(out), out

class s_cond_Critic(nn.Module):
    def __init__(self, n_layers, block_dim, latent_size, classes):
        super().__init__()
        self.label_embedding = nn.Embedding(classes, latent_size).requires_grad_(True)
        self.net = nn.Sequential(
            nn.Linear(latent_size*2, block_dim),
            nn.LeakyReLU(0.2, True),
            *[Block(block_dim) for _ in range(n_layers)]
        )
        self.end = nn.Linear(block_dim, 1)

    def forward(self, x, labels): 
        c = self.label_embedding(labels).squeeze(1)
        x = torch.cat((x,c),1)
        out = self.net(x)
        return self.end(out), out
