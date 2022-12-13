import torch
from torch import nn
from torch.nn import functional as F


class VariationalEncoder(nn.Module):

    def __init__(self,
                 input_size=784,
                 intermediate_size=512,
                 n_latent_dims=2) -> None:
        super(VariationalEncoder, self).__init__()

        self.input = nn.Linear(input_size, intermediate_size)
        self.latent_mu = nn.Linear(intermediate_size, n_latent_dims)
        self.latent_sigma = nn.Linear(intermediate_size, n_latent_dims)

        # q candidate distribution stats
        self.gaussian = torch.distributions.Normal(0, 1)
        self.gaussian.loc = self.gaussian.loc.cuda()  # get sampling on the GPU
        self.gaussian.scale = self.gaussian.scale.cuda()
        self.kl = 0

    def forward(self, x):
        # encode
        x = F.relu(self.input(x))

        mu = self.latent_mu(x)
        sigma = torch.exp(self.latent_sigma(x))

        assert torch.all(sigma >= 0)

        # reparameterization trick
        z = mu + sigma * self.gaussian.sample(mu.shape)

        self.kl = 0.5 * (1 + torch.log(sigma**2) - mu**2 - sigma**2).sum()

        return z, mu, sigma


class Decoder(nn.Module):

    def __init__(self,
                 n_latent_dims,
                 intermediate_size=512,
                 output_size=784) -> None:
        super(Decoder, self).__init__()

        self.latent_out = nn.Linear(n_latent_dims, intermediate_size)
        self.out = nn.Linear(intermediate_size, output_size)

    def forward(self, x):
        z = F.relu(self.latent_out(x))
        z = torch.sigmoid(self.out(z))

        return z


class VAE(nn.Module):

    def __init__(self,
                 n_latent_dims,
                 intermediate_size=512,
                 input_size=784) -> None:
        super(VAE, self).__init__()

        self.encoder = VariationalEncoder(input_size=input_size,
                                          intermediate_size=intermediate_size,
                                          n_latent_dims=n_latent_dims)
        self.decoder = Decoder(n_latent_dims=n_latent_dims,
                               intermediate_size=intermediate_size,
                               output_size=input_size)

    def forward(self, x) -> torch.Tensor:
        z, _, _ = self.encoder(x)
        return self.decoder(z)


class VAEForClassification(nn.Module):

    def __init__(self, n_latent_dims, n_out_dims, input_size=784) -> None:
        super(VAEForClassification, self).__init__()
        self.encoder = VariationalEncoder(input_size=input_size,
                                          n_latent_dims=n_latent_dims)
        self.decoder = nn.Linear(n_latent_dims, n_out_dims)

    def forward(self, x) -> torch.Tensor:
        z, _, _ = self.encoder(x)
        return self.decoder(z)


class SA_VAE(VAE):

    def __init__(self,
                 n_latent_dims,
                 intermediate_size=512,
                 input_size=784) -> None:
        super(SA_VAE, self).__init__(n_latent_dims, intermediate_size,
                                     input_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def sample(self, mu, sigma, rand=None):
        if rand is None:
            z = mu + sigma * self.encoder.gaussian.sample(mu.shape)
        else:
            rand.requires_grad = True
            z = mu + sigma * rand

        self.kl = 0.5 * (1 + torch.log(sigma**2) - mu**2 - sigma**2).sum()

        return z