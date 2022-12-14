import torch
from torch import nn
from torch.nn import functional as F
from blitz.modules import BayesianLinear


class VariationalEncoder(nn.Module):

    def __init__(self,
                 input_size=784,
                 intermediate_size=512,
                 n_latent_dims=2,
                 bayesian=False) -> None:
        super(VariationalEncoder, self).__init__()

        if bayesian:
            print("init bayesian encoder!")
            self.input = BayesianLinear(input_size, intermediate_size)
            self.latent_mu = BayesianLinear(intermediate_size, n_latent_dims)
            self.latent_sigma = BayesianLinear(intermediate_size,
                                               n_latent_dims)
        else:
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

        sigma_squared = torch.nan_to_num(sigma**2)
        assert torch.isinf(torch.log(sigma_squared)).any() == False

        self.kl = 0.5 * (1 + torch.log(sigma_squared) - mu**2 -
                         sigma_squared).sum()

        return z, mu, sigma


class Decoder(nn.Module):

    def __init__(self,
                 n_latent_dims,
                 intermediate_size=512,
                 output_size=784,
                 bayesian=False) -> None:
        super(Decoder, self).__init__()

        if bayesian:
            print("init bayesian decoder!")
            self.latent_out = BayesianLinear(n_latent_dims, intermediate_size)
            self.out = BayesianLinear(intermediate_size, output_size)
        else:
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
                 input_size=784,
                 bayesian_encoder=False,
                 bayesian_decoder=False) -> None:
        super(VAE, self).__init__()

        self.encoder = VariationalEncoder(input_size=input_size,
                                          intermediate_size=intermediate_size,
                                          n_latent_dims=n_latent_dims,
                                          bayesian=bayesian_encoder)
        self.decoder = Decoder(n_latent_dims=n_latent_dims,
                               intermediate_size=intermediate_size,
                               output_size=input_size,
                               bayesian=bayesian_decoder)

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
        assert torch.all(sigma >= 0)

        if rand is None:
            z = mu + sigma * self.encoder.gaussian.sample(mu.shape)
        else:
            rand.requires_grad = True
            z = mu + sigma * rand

        sigma_squared = torch.nan_to_num(sigma**2)
        assert torch.isinf(torch.log(sigma_squared)).any() == False
        self.kl = 0.5 * (1 + torch.log(sigma_squared) - mu**2 -
                         sigma_squared).sum()

        return z