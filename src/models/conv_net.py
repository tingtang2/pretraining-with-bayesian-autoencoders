from torch import nn
import torch
from bayesian_torch.layers import LinearReparameterization, Conv2dReparameterization, ConvTranspose2dReparameterization


# code from https://www.kaggle.com/code/maunish/training-vae-on-imagenet-pytorch
class ConvNetVAE(nn.Module):

    def __init__(self,
                 latent_dim=2,
                 bayesian_encoder=False,
                 bayesian_decoder=False):
        super().__init__()

        self.latent_dim = latent_dim
        self.shape = 26

        self.bayesian_encoder = bayesian_encoder
        self.bayesian_decoder = bayesian_decoder

        #encode
        if bayesian_encoder:
            print("init bayesian conv encoder")
            self.conv1 = Conv2dReparameterization(1,
                                                  32,
                                                  kernel_size=3,
                                                  stride=2)
            self.conv2 = Conv2dReparameterization(32,
                                                  64,
                                                  kernel_size=3,
                                                  stride=2)
            self.fc1 = LinearReparameterization(64 * (self.shape - 1)**2,
                                                2 * self.latent_dim)
        else:
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
            self.fc1 = nn.Linear(64 * (self.shape - 1)**2, 2 * self.latent_dim)

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.scale = nn.Parameter(torch.tensor([0.0]))

        self.eps = 1e-8
        self.criterion = nn.MSELoss(reduction='sum')

        #decode
        if bayesian_decoder:
            print("init bayesian conv decoder")
            self.fc2 = LinearReparameterization(self.latent_dim,
                                                (self.shape**2) * 32)
            self.conv3 = ConvTranspose2dReparameterization(32,
                                                           64,
                                                           kernel_size=2,
                                                           stride=2)
            self.conv4 = ConvTranspose2dReparameterization(64,
                                                           32,
                                                           kernel_size=2,
                                                           stride=2)
            self.conv5 = ConvTranspose2dReparameterization(32,
                                                           1,
                                                           kernel_size=1,
                                                           stride=1)
        else:
            self.fc2 = nn.Linear(self.latent_dim, (self.shape**2) * 32)
            self.conv3 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
            self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.conv5 = nn.ConvTranspose2d(32, 1, kernel_size=1, stride=1)

        self.decoder = nn.ModuleList(
            [self.fc2, self.conv3, self.conv4, self.conv5])

    def encode(self, x):
        if self.bayesian_encoder:
            x = self.relu(self.conv1(x, return_kl=False))
            x = self.relu(self.conv2(x, return_kl=False))
            x = self.relu(self.flatten(x, return_kl=False))
            x = self.fc1(x, return_kl=False)
        else:
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.flatten(x))
            x = self.fc1(x)

        mean, logvar = torch.split(x, self.latent_dim, dim=1)
        return mean, logvar

    def decode(self, eps):
        if self.bayesian_decoder:
            x = self.relu(self.fc2(eps, return_kl=False))
            x = torch.reshape(x, (x.shape[0], 32, self.shape, self.shape))
            x = self.relu(self.conv3(x, return_kl=False))
            x = self.relu(self.conv4(x, return_kl=False))
            x = self.conv5(x, return_kl=False)
        else:
            x = self.relu(self.fc2(eps))
            x = torch.reshape(x, (x.shape[0], 32, self.shape, self.shape))
            x = self.relu(self.conv3(x))
            x = self.relu(self.conv4(x))
            x = self.conv5(x)
        return x

    def reparamatrize(self, mean, std):
        std = torch.nan_to_num(std, nan=.01, neginf=.01)
        std = torch.relu(std) + self.eps
        if not torch.all(std > 0):
            print(std)
        assert torch.all(std > 0)
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

    def new_kl_loss(self, mean, std):
        sigma_squared = torch.nan_to_num(std**2, nan=.01, neginf=.01)
        assert torch.all(sigma_squared > 0)
        assert torch.isinf(torch.log(sigma_squared)).any() == False

        return 0.5 * (1 + torch.log(sigma_squared) - mean**2 -
                      sigma_squared).sum()

    def new_gaussian_likelihood(self, inputs, outputs):
        return self.criterion(outputs.reshape((outputs.size(0), -1)),
                              inputs.reshape((inputs.size(0), -1)))

    def gaussian_likelihood(self, inputs, outputs, scale):
        dist = torch.distributions.Normal(outputs, torch.exp(scale))
        log_pxz = dist.log_prob(inputs)
        return log_pxz.sum(dim=(1, 2, 3))

    def loss_fn(self, inputs, outputs, z, mean, std):
        kl_loss = self.kl_loss(z, mean, std)
        rec_loss = self.gaussian_likelihood(inputs, outputs, self.scale)

        return torch.sum(kl_loss - rec_loss), rec_loss.sum()

    def new_loss_fn(self, z, inputs, outputs, mean, std):
        kl_loss = self.new_kl_loss(mean, std)
        # kl_loss = self.kl_loss(z, mean, std)
        rec_loss = self.new_gaussian_likelihood(inputs, outputs)

        return torch.sum(-kl_loss + rec_loss), rec_loss.sum()

    def forward(self, inputs):
        mean, logvar = self.encode(inputs)
        logvar = torch.nan_to_num(logvar, nan=.01, neginf=.01)

        std = torch.exp(logvar / 2) + self.eps
        assert torch.all(std > 0)
        z = self.reparamatrize(mean, std)
        outputs = self.decode(z)

        # loss = self.loss_fn(inputs, outputs, z, mean, std)
        loss = self.new_loss_fn(z, inputs, outputs, mean, std)
        return loss, (outputs, z, mean, std)
