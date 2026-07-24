import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -------------------------
# Generator and Discriminator Models
# -------------------------
class Generator(nn.Module):
    """Simple CNN Generator"""
    def __init__(self, latent_dim: int = 100, hidden_dim: int = 128, 
                 output_channels: int = 1, img_size: int = 32, use_bn: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # Calculate initial size based on architecture
        init_size = img_size // 4
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * init_size * init_size),
            nn.BatchNorm1d(hidden_dim * init_size * init_size) if use_bn else nn.Identity(),
            nn.ReLU(True)
        )
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2) if use_bn else nn.Identity(),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim // 4) if use_bn else nn.Identity(),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(hidden_dim // 4, output_channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        x = self.fc(z)
        x = x.view(batch_size, -1, self.img_size // 4, self.img_size // 4)
        x = self.deconv(x)
        return x


class Discriminator(nn.Module):
    """Simple CNN Discriminator"""
    def __init__(self, input_channels: int = 1, hidden_dim: int = 128, 
                 img_size: int = 32, use_bn: bool = True):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim // 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(hidden_dim // 2, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Calculate flattened size
        self.flatten_size = hidden_dim * (img_size // 8) * (img_size // 8)
        
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 1),
            nn.Sigmoid()  # Using BCELoss
        )
        
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.conv(img)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x