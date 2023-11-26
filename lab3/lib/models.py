import torch.nn as nn

class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding),
            nn.BatchNorm2d(out_dim),
        )
    
    def forward(self, x):
        out = self.layers(x)
        return out


class Block(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, stride, padding),
            nn.BatchNorm2d(dim),
            nn.ReLU(dim),
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.BatchNorm2d(dim),
        )
        
    def forward(self, x):
        out = x + self.layers(x)
        return out


class ResNet(nn.Module):
    def __init__(self, blocks, dims, dropout, num_classes):
        super().__init__()
        self.downsamples = nn.ModuleList()
        for i in range(4):
            if i == 0:
                self.downsamples.append(Downsample(3, dims[0], 3, 1, 1))
            else:
                self.downsamples.append(Downsample(dims[i-1], dims[i], 2, 2, 0))
        
        self.layers = nn.ModuleList()
        for i in range(4):
            layers = nn.Sequential()
            for _ in range(blocks[i]):
                layers.append(Block(dims[i], 3, 1, 1))
            self.layers.append(layers)

        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.BatchNorm1d(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
    
    def forward(self, x):
        for i in range(4):
            x = self.downsamples[i](x)
            x = self.layers[i](x)
        x = self.dropout(self.norm(x.mean([-1, -2])))
        out = self.head(x)
        return out