from torch import nn
import torch


# code from https://www.kaggle.com/code/maunish/training-vae-on-imagenet-pytorch
class ConvNetVAE(nn.Module):

    def __init__(self, latent_dim=2):
        super().__init__()

        self.latent_dim = latent_dim
        self.shape = 26

        #encode
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*(self.shape-1)**2, 2 * self.latent_dim)
        self.relu = nn.ReLU()
        self.scale = nn.Parameter(torch.tensor([0.0]))

        #decode
        self.fc2 = nn.Linear(self.latent_dim, (self.shape**2) * 32)
        self.conv3 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv5 = nn.ConvTranspose2d(32, 1, kernel_size=1, stride=1)

    def encode(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.flatten(x))
        x = self.fc1(x)
        mean, logvar = torch.split(x, self.latent_dim, dim=1)
        return mean, logvar

    def decode(self, eps):
        x = self.relu(self.fc2(eps))
        x = torch.reshape(x, (x.shape[0], 32, self.shape, self.shape))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x

    def reparamatrize(self, mean, std):
        q = torch.distributions.Normal(mean, std)
        return q.rsample()

    def kl_loss(self, z, mean, std):
        p = torch.distributions.Normal(torch.zeros_like(mean),
                                       torch.ones_like(std))
        q = torch.distributions.Normal(mean, torch.exp(std / 2))

        log_pz = p.log_prob(z)
        log_qzx = q.log_prob(z)

        kl_loss = (log_qzx - log_pz)
        kl_loss = kl_loss.sum(-1)
        return kl_loss

    def gaussian_likelihood(self, inputs, outputs, scale):
        dist = torch.distributions.Normal(outputs, torch.exp(scale))
        log_pxz = dist.log_prob(inputs)
        return log_pxz.sum(dim=(1, 2, 3))

    def loss_fn(self, inputs, outputs, z, mean, std):
        kl_loss = self.kl_loss(z, mean, std)
        rec_loss = self.gaussian_likelihood(inputs, outputs, self.scale)

        return torch.sum(kl_loss - rec_loss)

    def forward(self, inputs):
        mean, logvar = self.encode(inputs)
        std = torch.exp(logvar / 2)
        z = self.reparamatrize(mean, std)
        outputs = self.decode(z)

        loss = self.loss_fn(inputs, outputs, z, mean, std)
        return loss, (outputs, z, mean, std)
