import torch
from torch.utils.data import DataLoader

from models.vae import VAE, VAEForClassification
from trainers.base_trainer import BaseTrainer


class VAEColorTrainer(BaseTrainer):

    def __init__(self, **kwargs) -> None:
        super(VAEColorTrainer, self).__init__(**kwargs)

        self.model = VAE(n_latent_dims=2).to(self.device)
        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate,
                                             amsgrad=True)
        self.x_dim = 3072

    def train(self, loader: DataLoader):
        self.model.train()
        running_loss = 0.0

        for i, (x, y) in enumerate(loader):
            self.optimizer.zero_grad()

            x = x.reshape(x.size(0), self.x_dim).to(self.device)
            x_hat = self.model(x)
            loss = self.criterion(x_hat, x) - self.model.encoder.kl

            loss.backward()
            running_loss += loss.item()

            self.optimizer.step()

        return running_loss / (len(loader) * loader.batch_size)

    def eval(self, loader: DataLoader) -> float:
        predictive_ELBO = 0.0

        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x = x.reshape(x.size(0), self.x_dim).to(self.device)
                x_hat = self.model(x)
                ELBO_batch = -self.criterion(x_hat, x) + self.model.encoder.kl

                predictive_ELBO += ELBO_batch.item()

        return predictive_ELBO / (len(loader) * loader.batch_size)