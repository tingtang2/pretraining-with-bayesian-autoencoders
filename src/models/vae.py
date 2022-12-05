import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):

    def __init__(self, n_latent_dims) -> None:
        super(VAE, self).__init__()

        # encoder
        self.input = nn.Linear(784, 512)
        self.latent_mu = nn.Linear(512, n_latent_dims)
        self.latent_sigma = nn.Linear(512, n_latent_dims)

        # q candidate distribution stats
        self.gaussian = torch.distributions.Normal(0, 1)
        self.gaussian.loc = self.gaussian.loc.cuda()  # get sampling on the GPU
        self.gaussian.scale = self.gaussian.scale.cuda()
        self.kl = 0

        # decoder
        self.latent_out = nn.Linear(n_latent_dims, 512)
        self.out = nn.Linear(512, 784)

    def forward(self, x, encode=False, decode=False) -> torch.Tensor:
        if decode:
            z = x
        else:
            # encode
            x = F.relu(self.input(x))

            mu = self.latent_mu(x)
            sigma = torch.exp(self.latent_sigma(x))

            assert torch.all(sigma >= 0)

            # reparameterization trick
            z = mu + sigma * self.gaussian.sample(mu.shape)

            if encode:
                return z, mu, sigma

            self.kl = 0.5 * (1 + torch.log(sigma**2) - mu**2 - sigma**2).sum()

        # decode
        z = F.relu(self.latent_out(z))
        z = torch.sigmoid(self.out(z))

        return z