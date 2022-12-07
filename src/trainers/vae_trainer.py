from typing import Tuple

import numpy as np
import torch

from trainers.base_trainer import BaseTrainer
from sklearn.model_selection import train_test_split
import torchvision
from torch.utils.data import DataLoader

from data import notMNISTDataset
from models.vae import VAE

from tqdm import trange


class VAETrainer(BaseTrainer):

    def __init__(self, **kwargs) -> None:
        super(VAETrainer, self).__init__(**kwargs)

        self.model = VAE(n_latent_dims=2).to(self.device)
        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate,
                                             amsgrad=True)

    def pretrain(self):
        train_loader, valid_loader = self.create_pretraining_dataloaders()

        training_elbo = []
        predictive_elbo = []

        for i in trange(self.epochs):
            training_elbo.append(-self.train(train_loader))
            predictive_elbo.append(self.eval(valid_loader))
            print(
                f'ELBO: {training_elbo[-1]}, predictive ELBO: {predictive_elbo[-1]}'
            )

    def train(self, loader):
        self.model.train()
        running_loss = 0.0

        for i, (x, y) in enumerate(loader):
            self.optimizer.zero_grad()

            x_hat = self.model(x)
            loss = self.criterion(x_hat, x) - self.model.kl

            loss.backward()
            running_loss += loss.item()

            self.optimizer.step()

        return running_loss / (len(loader) * loader.batch_size)

    def eval(self, loader) -> float:
        predictive_ELBO = 0.0

        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x_hat = self.model(x)
                ELBO_batch = -self.criterion(x_hat, x) + self.model.kl

                predictive_ELBO += ELBO_batch.item()

        return predictive_ELBO / (len(loader) * loader.batch_size)


class VAENotMNIST2MNISTTrainer(VAETrainer):

    def create_pretraining_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        img_data = np.load(self.data_dir + 'notMNIST_small/images.npy')
        label_data = np.load(self.data_dir +
                             'notMNIST_small/labels_singular.npy')

        X_train, X_val, y_train, y_val = train_test_split(
            img_data, label_data, test_size=0.20, random_state=self.seed)
        print(f'train set size: {X_train.shape}, val set size: {X_val.shape}')

        # set up dataset objects
        train_dataset = notMNISTDataset(
            torch.from_numpy(X_train.astype(np.float32)).to(self.device),
            torch.from_numpy(y_train).to(self.device))
        valid_dataset = notMNISTDataset(
            torch.from_numpy(X_val.astype(np.float32)).to(self.device),
            torch.from_numpy(y_val).to(self.device))

        # set up data loaders
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=len(valid_dataset),
                                  shuffle=False)

        return train_loader, valid_loader

    def create_finetuning_dataloaders(self):
        MNIST_data = torchvision.datasets.MNIST('/home/tingchen/data/',
                                                download=True)

    def finetune():
        pass