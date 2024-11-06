import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['DMCount', 'vgg19']
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class GlobalAttention(nn.Module):
    def __init__(self, in_dim):
        super(GlobalAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

class LocalAttention(nn.Module):
    def __init__(self, in_dim, kernel_size=7):
        super(LocalAttention, self).__init__()
        self.conv = nn.Conv2d(in_dim, in_dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = self.conv(x)
        out = self.gamma * out + x
        return out

class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features

    def forward(self, x):
        return self.features(x)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def vgg19():
    """VGG 19-layer model (configuration "E") pre-trained on ImageNet."""
    model = VGG(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model

class DMCount(nn.Module):
    def __init__(self, base_model):
        super(DMCount, self).__init__()
        self.base_model = base_model
        
        # Ensure the feature map dimension (in_dim) aligns with the output of VGG backbone
        in_dim = 512  # Output channels from VGG-19 backbone after feature extraction

        # Attention modules
        self.global_attention = GlobalAttention(in_dim=in_dim)
        self.local_attention = LocalAttention(in_dim=in_dim)
        
        # Density map prediction head
        self.reg_layer = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        x = self.base_model(x)
        
        # Upsample to 1/8 of the input image size
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Apply global and local attention
        x = self.global_attention(x)
        x = self.local_attention(x)

        # Pass through the density map head
        x = self.reg_layer(x)
        density_map = self.density_layer(x)

        # Apply softmax normalization for OT compatibility
        density_map_normed = F.softmax(density_map.view(x.size(0), -1), dim=1).view_as(density_map)

        return density_map, density_map_normed
