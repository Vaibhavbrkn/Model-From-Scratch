import torch
import torch.nn as nn
from functools import partial


class PatchEmbed(nn.Module):

    def __init__(self, img_size, patch_size, in_dim=3, embed_dim=512, norm_layer=None):
        super().__init__()

        self.conv = nn.Conv2d(
            in_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.pat = img_size // patch_size * img_size // patch_size

    def forward(self, img):

        x = self.conv(img)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Mixer(nn.Module):
    def __init__(self, dim, token_dim, channel_dim, patch, drop=0., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.token_mix = MLP(patch, token_dim, drop)
        self.channel_mix = MLP(dim, channel_dim, drop)

    def forward(self, x):
        x = x + self.token_mix(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_mix(self.norm2(x))
        return x


class MlpMixer(nn.Module):
    def __init__(self, dim, token_dim, channel_dim, img_size, patch_size, classes=10, num_blocks=8, drop=0., norm=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch_size,
                                norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.mixer = nn.Sequential(*[
            Mixer(dim, token_dim, channel_dim, self.patch.pat, drop)
            for _ in range(num_blocks)
        ])
        self.norm = norm(dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, x):
        x = self.patch(x)
        x = self.mixer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


if __name__ == "__main__":
    img = torch.randn(1, 3, 224, 224)
    ML = MlpMixer(512, 256, 2048, 224, 16)
    x = ML(img)
    print(x.shape)
