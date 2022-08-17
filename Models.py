import torch
import torch.nn as nn
from torchvision.models import resnet18


class Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.expander = nn.Sequential(
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),

            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),

            nn.Linear(2048, 2048)
        )

    def forward(self, x):
        return self.expander(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(pretrained=True)
        self.encoder = nn.Sequential(*(list(backbone.children())[:-1]))

    def forward(self, x):
        bs, _, _, _ = x.size()
        x = self.encoder(x)
        x = x.view(bs, 512)
        return x


class Decoder(nn.Module):
    def __init__(self, reshaped_dim=(64, 32, 32)):
        super().__init__()

        self.latent_dim = 512
        self.reshaped_dim = reshaped_dim
        reshaped_dim_flatten = self.reshaped_dim[0] * self.reshaped_dim[1] * self.reshaped_dim[2]

        self.latent_manipulation = nn.Sequential(
            nn.Linear(self.latent_dim, reshaped_dim_flatten),
            nn.ReLU(True)
        )

        self.latent_2_image = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(self.reshaped_dim[0], self.reshaped_dim[0] * 2, kernel_size=4, padding="same"),
            nn.ReLU(True),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(self.reshaped_dim[0] * 2, self.reshaped_dim[0], kernel_size=4, padding="same"),
            nn.ReLU(True),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(self.reshaped_dim[0], self.reshaped_dim[0], kernel_size=4, padding="same"),
            nn.ReLU(True),

            nn.Conv2d(self.reshaped_dim[0], 3, kernel_size=7, stride=2, padding=3),
            nn.Tanh()
        )

    def forward(self, z):
        batch_size, _ = z.size()
        new_z = self.latent_manipulation(z)
        new_z = new_z.view([batch_size] + list(self.reshaped_dim))
        return self.latent_2_image(new_z)


if __name__ == '__main__':
    enc = Encoder()
    proj = Projector()
    x = torch.rand((5, 3, 224, 224))
    z = enc(x)
    zz = proj(z)
    print()

    dec = Decoder()
    z = torch.rand((5, 512))
    x_ = dec(z)

    print()
