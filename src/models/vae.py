import torch
from torch import nn
from torch.nn import functional as F
# from blitz.modules import BayesianLinear
from bayesian_torch.layers import LinearReparameterization


class VariationalEncoder(nn.Module):

    def __init__(self,
                 input_size=784,
                 intermediate_size=512,
                 n_latent_dims=2,
                 bayesian=False,
                 num_extra_layers=0) -> None:
        super(VariationalEncoder, self).__init__()

        self.num_extra_layers = num_extra_layers
        # self.extra_layers = {}

        if bayesian:
            print("init bayesian encoder! with: {} latent dims".format(
                n_latent_dims))
            # self.input = BayesianLinear(input_size, intermediate_size, bias=False, posterior_mu_init=0.1)
            # self.latent_mu = BayesianLinear(intermediate_size, n_latent_dims, bias=False, posterior_mu_init=0.1)
            # self.latent_sigma = BayesianLinear(intermediate_size, n_latent_dims, bias=False, posterior_mu_init=0.1)
            self.input = LinearReparameterization(input_size, intermediate_size) #, prior_mean=1, posterior_mu_init=1, posterior_rho_init=3
            self.extra_layers = nn.ModuleList([LinearReparameterization(intermediate_size, intermediate_size) for _ in range(num_extra_layers)])
            # for i in range(num_extra_layers):
            #     self.extra_layers[i] = LinearReparameterization(intermediate_size, intermediate_size)
            self.latent_mu = LinearReparameterization(intermediate_size, n_latent_dims)
            self.latent_sigma = LinearReparameterization(intermediate_size, n_latent_dims)
        else:
            self.input = nn.Linear(input_size, intermediate_size)
            self.extra_layers = nn.ModuleList([nn.Linear(intermediate_size, intermediate_size) for _ in range(num_extra_layers)])
            # for i in range(num_extra_layers):
            #     self.extra_layers[i] = nn.Linear(intermediate_size, intermediate_size)
            self.latent_mu = nn.Linear(intermediate_size, n_latent_dims)
            self.latent_sigma = nn.Linear(intermediate_size, n_latent_dims)

        # q candidate distribution stats
        self.gaussian = torch.distributions.Normal(0, 1)
        self.gaussian.loc = self.gaussian.loc.cuda()  # get sampling on the GPU
        self.gaussian.scale = self.gaussian.scale.cuda()
        self.kl = 0
        self.bayesian = bayesian

    def forward(self, x):
        if self.bayesian:
            out = self.input(x, return_kl=False)
            x = F.relu(out)
            for i in range(self.num_extra_layers):
                x = self.extra_layers[i](x, return_kl=False)
            mu = self.latent_mu(x, return_kl=False)
            sigma = torch.exp(self.latent_sigma(x, return_kl=False))
        else:
            x = F.relu(self.input(x))
            for i in range(self.num_extra_layers):
                x = self.extra_layers[i](x)
            mu = self.latent_mu(x)
            sigma = torch.exp(self.latent_sigma(x))

        sigma = torch.nan_to_num(sigma, nan=.01, neginf=.01)
        assert torch.all(sigma > 0)

        # reparameterization trick
        z = mu + sigma * self.gaussian.sample(mu.shape)

        sigma_squared = torch.nan_to_num(sigma**2, nan=.01, neginf=.01)
        assert torch.all(sigma_squared > 0)
        assert torch.isinf(torch.log(sigma_squared)).any() == False

        self.kl = 0.5 * (1 + torch.log(sigma_squared) - mu**2 -
                         sigma_squared).sum()

        return z, mu, sigma


class Decoder(nn.Module):

    def __init__(self,
                 n_latent_dims,
                 intermediate_size=512,
                 output_size=784,
                 bayesian=False,
                 num_extra_layers=0) -> None:
        super(Decoder, self).__init__()
        self.bayesian = bayesian
        self.num_extra_layers = num_extra_layers
        self.extra_layers = {}

        if bayesian:
            print("init bayesian decoder!")
            self.latent_out = LinearReparameterization(n_latent_dims, intermediate_size)
            self.extra_layers = nn.ModuleList([LinearReparameterization(intermediate_size, intermediate_size) for _ in range(num_extra_layers)])
            # for i in range(num_extra_layers):
            #     self.extra_layers[i] = LinearReparameterization(intermediate_size, intermediate_size)
            self.out = LinearReparameterization(intermediate_size, output_size)
        else:
            self.latent_out = nn.Linear(n_latent_dims, intermediate_size)
            self.extra_layers = nn.ModuleList([nn.Linear(intermediate_size, intermediate_size) for _ in range(num_extra_layers)])
            # for i in range(num_extra_layers):
            #     self.extra_layers[i] = nn.Linear(intermediate_size, intermediate_size)
            self.out = nn.Linear(intermediate_size, output_size)

    def forward(self, x):
        if self.bayesian:
            z = F.relu(self.latent_out(x, return_kl=False))
            for i in range(self.num_extra_layers):
                z = self.extra_layers[i](z, return_kl=False)
            z = torch.sigmoid(self.out(z, return_kl=False))
        else:
            z = F.relu(self.latent_out(x))
            for i in range(self.num_extra_layers):
                z = self.extra_layers[i](z)
            z = torch.sigmoid(self.out(z))

        return z


class VAE(nn.Module):

    def __init__(self,
                 n_latent_dims,
                 intermediate_size=512,
                 input_size=784,
                 bayesian_encoder=False,
                 bayesian_decoder=False,
                 num_extra_layers=0) -> None:
        super(VAE, self).__init__()

        self.encoder = VariationalEncoder(input_size=input_size,
                                          intermediate_size=intermediate_size,
                                          n_latent_dims=n_latent_dims,
                                          bayesian=bayesian_encoder,
                                          num_extra_layers=num_extra_layers)
        self.decoder = Decoder(n_latent_dims=n_latent_dims,
                               intermediate_size=intermediate_size,
                               output_size=input_size,
                               bayesian=bayesian_decoder,
                               num_extra_layers=num_extra_layers)

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
        self.epsilon = 1e-08

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def sample(self, mu, sigma, rand=None):
        sigma = torch.relu(sigma) + self.epsilon
        # print(sigma)
        assert torch.all(sigma > 0)

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