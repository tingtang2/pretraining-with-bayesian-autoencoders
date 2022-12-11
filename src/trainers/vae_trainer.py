import logging
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from data import notMNISTDataset
from models.vae import VAE, VAEForClassification
from trainers.base_trainer import BaseTrainer


class VAETrainer(BaseTrainer):

    def __init__(self, **kwargs) -> None:
        super(VAETrainer, self).__init__(**kwargs)

        self.model = VAE(n_latent_dims=2).to(self.device)
        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate,
                                             amsgrad=True)

    def train(self, loader: DataLoader):
        self.model.train()
        running_loss = 0.0

        for i, (x, y) in enumerate(loader):
            self.optimizer.zero_grad()

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
                x_hat = self.model(x)
                ELBO_batch = -self.criterion(x_hat, x) + self.model.encoder.kl

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
        transform = transforms.Compose([transforms.ToTensor()])
        MNIST_data_train = torchvision.datasets.MNIST(self.data_dir,
                                                      train=True,
                                                      transform=transform,
                                                      download=False)

        train_set, val_set = torch.utils.data.random_split(
            MNIST_data_train, [50000, 10000])
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            val_set, batch_size=len(MNIST_data_train), shuffle=False)
        return train_loader, valid_loader

    def pretrain(self):
        train_loader, valid_loader = self.create_pretraining_dataloaders()

        training_elbo = []
        predictive_elbo = []

        for i in trange(self.pretrain_epochs):
            training_elbo.append(-self.train(train_loader))
            predictive_elbo.append(self.eval(valid_loader))
            logging.info(
                f'epoch: {i} ELBO: {training_elbo[-1]}, predictive ELBO: {predictive_elbo[-1]}'
            )

        name = 'vae_pretrained_notmnist'
        self.save_model(name=name)
        self.plot_latent(loader=train_loader, name=name)
        self.plot_reconstructed(name=name)
        self.save_metrics(training_elbo,
                          name=name + '_training_elbo',
                          phase='pretrain')
        self.save_metrics(predictive_elbo,
                          name=name + '_predictive_elbo',
                          phase='pretrain')

    def finetune_train(self, loader: DataLoader):
        self.model.train()
        running_loss = 0.0

        for i, (x, y) in enumerate(loader):
            self.optimizer.zero_grad()

            reshaped_x = x.reshape(x.size(0), 784)

            y_hat = self.model(reshaped_x.to(self.device))
            # TODO: see what happens when we keep self.model.kl (it's really bad!)
            loss = self.criterion(y_hat,
                                  y.to(self.device))  # - self.model.encoder.kl

            loss.backward()
            running_loss += loss.item()

            self.optimizer.step()

        return running_loss / len(loader)

    def finetune_eval(self, loader: DataLoader):
        num_right = 0

        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                reshaped_x = x.reshape(x.size(0), 784)
                y_hat = self.model(reshaped_x.to(self.device))
                num_right += torch.sum(
                    y.to(self.device) == torch.argmax(
                        y_hat, dim=-1)).detach().cpu().item()

        return num_right / len(loader.dataset)

    def finetune(self):
        train_loader, valid_loader = self.create_finetuning_dataloaders()
        self.model = VAE(n_latent_dims=2).to(self.device)
        self.model.load_state_dict(
            torch.load(f'{self.save_dir}models/vae_pretrained_notmnist.pt'))

        # model surgery for image classification
        self.old_decoder = self.model.decoder
        self.model.decoder = nn.Linear(2, 10).to(self.device)
        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate,
                                             amsgrad=True)
        self.criterion = nn.CrossEntropyLoss()

        training_loss = []
        val_loss = []
        training_accuracy = []
        val_accuracy = []

        for i in trange(self.finetune_epochs):
            training_loss.append(self.finetune_train(train_loader))
            val_loss.append(self.finetune_train(valid_loader))
            training_accuracy.append(self.finetune_eval(train_loader))
            val_accuracy.append(self.finetune_eval(valid_loader))

            logging.info(
                f'epoch: {i} training loss: {training_loss[-1]:.3f} val loss:{val_loss[-1]:.3f} training accuracy: {training_accuracy[-1]:.3f} val acc: {val_accuracy[-1]:.3f}'
            )

        name = 'vae_pretrained_notmnist_finetune_mnist'
        # self.save_model(name=name)
        # self.plot_latent_from_fine(loader=train_loader, name=name)
        # self.plot_reconstructed_from_fine(name=name)
        # self.save_metrics(training_loss,
        #                   name=name + '_train_loss',
        #                   phase='finetune')
        # self.save_metrics(val_accuracy,
        #                   name=name + '_val_accuracy',
        #                   phase='finetune')

    def plot_latent_from_fine(self, loader, name: str):
        self.model.eval()

        labels = []
        z_s = []
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                if i > 1000:
                    break
                z, mu, sigma = self.model.encoder(
                    x.reshape((x.size(0), 784)).to(self.device))
                z_s.append(z.cpu().detach().numpy())

                labels += y.cpu().numpy().tolist()

        z_s = np.vstack(z_s)
        pd_dict = {
            'first dimension': z_s[:, 0],
            'second dimension': z_s[:, 1],
            'labels': labels
        }

        df = pd.DataFrame(pd_dict)
        df['labels'] = df['labels'].astype(str)

        fig = px.scatter(
            df,
            x='first dimension',
            y='second dimension',
            color='labels',
            title='Training set projected into learned latent space')
        fig.write_html(self.save_dir + f'plots/{name}_latent_space.html')
        fig.write_image(self.save_dir + f'plots/{name}_latent_space.png')


class VAENoPretrainingMNIST(VAENotMNIST2MNISTTrainer):

    def finetune(self):
        train_loader, valid_loader = self.create_finetuning_dataloaders()
        self.model = VAEForClassification(n_latent_dims=2,
                                          n_out_dims=10).to(self.device)

        # model surgery for image classification
        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate,
                                             amsgrad=True)
        self.criterion = nn.CrossEntropyLoss()

        training_loss = []
        val_loss = []
        training_accuracy = []
        val_accuracy = []

        for i in trange(self.finetune_epochs):
            training_loss.append(self.finetune_train(train_loader))
            val_loss.append(self.finetune_train(valid_loader))
            training_accuracy.append(self.finetune_eval(train_loader))
            val_accuracy.append(self.finetune_eval(valid_loader))

            logging.info(
                f'epoch: {i} training loss: {training_loss[-1]:.3f} val loss:{val_loss[-1]:.3f} training accuracy: {training_accuracy[-1]:.3f} val acc: {val_accuracy[-1]:.3f}'
            )

        name = 'vae_no_pretraining_mnist'
        #self.save_model(name=name)


class VAENoPretrainingFashionMNIST(VAENotMNIST2MNISTTrainer):

    def create_finetuning_dataloaders(self):
        transform = transforms.Compose([transforms.ToTensor()])
        MNIST_data_train = torchvision.datasets.FashionMNIST(
            self.data_dir, train=True, transform=transform, download=False)

        train_set, val_set = torch.utils.data.random_split(
            MNIST_data_train, [50000, 10000])
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            val_set, batch_size=len(MNIST_data_train), shuffle=False)
        return train_loader, valid_loader

    def finetune(self):
        train_loader, valid_loader = self.create_finetuning_dataloaders()
        self.model = VAEForClassification(n_latent_dims=2,
                                          n_out_dims=10).to(self.device)

        # model surgery for image classification
        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate,
                                             amsgrad=True)
        self.criterion = nn.CrossEntropyLoss()

        training_loss = []
        val_loss = []
        training_accuracy = []
        val_accuracy = []

        for i in trange(self.finetune_epochs):
            training_loss.append(self.finetune_train(train_loader))
            val_loss.append(self.finetune_train(valid_loader))
            training_accuracy.append(self.finetune_eval(train_loader))
            val_accuracy.append(self.finetune_eval(valid_loader))

            logging.info(
                f'epoch: {i} training loss: {training_loss[-1]:.3f} val loss:{val_loss[-1]:.3f} training accuracy: {training_accuracy[-1]:.3f} val acc: {val_accuracy[-1]:.3f}'
            )

        name = 'vae_no_pretraining_fashion_mnist'
        #self.save_model(name=name)


class VAENotMNIST2FashionMNISTTrainer(VAENoPretrainingFashionMNIST):

    def finetune(self):
        train_loader, valid_loader = self.create_finetuning_dataloaders()
        self.model = VAE(n_latent_dims=2).to(self.device)
        self.model.load_state_dict(
            torch.load(f'{self.save_dir}models/vae_pretrained_notmnist.pt'))

        # model surgery for image classification
        self.old_decoder = self.model.decoder
        self.model.decoder = nn.Linear(2, 10).to(self.device)
        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate,
                                             amsgrad=True)
        self.criterion = nn.CrossEntropyLoss()

        training_loss = []
        val_loss = []
        training_accuracy = []
        val_accuracy = []

        for i in trange(self.finetune_epochs):
            training_loss.append(self.finetune_train(train_loader))
            val_loss.append(self.finetune_train(valid_loader))
            training_accuracy.append(self.finetune_eval(train_loader))
            val_accuracy.append(self.finetune_eval(valid_loader))

            logging.info(
                f'epoch: {i} training loss: {training_loss[-1]:.3f} val loss:{val_loss[-1]:.3f} training accuracy: {training_accuracy[-1]:.3f} val acc: {val_accuracy[-1]:.3f}'
            )

        name = 'vae_pretrained_notmnist_finetune_mnist'


##################################################################

# Pretraining on Fashion MNIST


class VAEFashionMNIST2notMNISTTrainer(VAENotMNIST2FashionMNISTTrainer):

    def create_pretraining_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        return super().create_finetuning_dataloaders()

    def create_finetuning_dataloaders(self):
        return super().create_pretraining_dataloaders()

    def pretrain(self):
        train_loader, valid_loader = self.create_pretraining_dataloaders()

        training_elbo = []
        predictive_elbo = []

        for i in trange(self.pretrain_epochs):
            training_elbo.append(-self.train(train_loader))
            predictive_elbo.append(self.eval(valid_loader))
            logging.info(
                f'epoch: {i} ELBO: {training_elbo[-1]}, predictive ELBO: {predictive_elbo[-1]}'
            )
        name = 'vae_pretrained_fashion_mnist'
        self.save_model(name=name)
        self.plot_latent_from_fine(loader=train_loader, name=name)
        self.plot_reconstructed(name=name)
        # self.save_metrics(training_elbo,
        #                   name=name + '_training_elbo',
        #                   phase='pretrain')
        # self.save_metrics(predictive_elbo,
        #                   name=name + '_predictive_elbo',
        #                   phase='pretrain')

    def finetune(self):
        train_loader, valid_loader = self.create_finetuning_dataloaders()
        self.model = VAE(n_latent_dims=2).to(self.device)
        self.model.load_state_dict(
            torch.load(
                f'{self.save_dir}models/vae_pretrained_fashion_mnist.pt'))

        # model surgery for image classification
        self.old_decoder = self.model.decoder
        self.model.decoder = nn.Linear(2, 10).to(self.device)
        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate,
                                             amsgrad=True)
        self.criterion = nn.CrossEntropyLoss()

        training_loss = []
        val_loss = []
        training_accuracy = []
        val_accuracy = []

        for i in trange(self.finetune_epochs):
            training_loss.append(self.finetune_train(train_loader))
            val_loss.append(self.finetune_train(valid_loader))
            training_accuracy.append(self.finetune_eval(train_loader))
            val_accuracy.append(self.finetune_eval(valid_loader))

            logging.info(
                f'epoch: {i} training loss: {training_loss[-1]:.3f} val loss:{val_loss[-1]:.3f} training accuracy: {training_accuracy[-1]:.3f} val acc: {val_accuracy[-1]:.3f}'
            )

        name = 'vae_pretrained_notmnist_finetune_mnist'
        # self.save_model(name=name)
        # self.plot_latent_from_fine(loader=train_loader, name=name)
        # self.plot_reconstructed_from_fine(name=name)
        # self.save_metrics(training_loss,
        #                   name=name + '_train_loss',
        #                   phase='finetune')
        # self.save_metrics(val_accuracy,
        #                   name=name + '_val_accuracy',
        #                   phase='finetune')

    def train(self, loader: DataLoader):
        self.model.train()
        running_loss = 0.0

        for i, (x, y) in enumerate(loader):
            self.optimizer.zero_grad()

            x_hat = self.model(x.reshape((x.size(0), 784)).to(self.device))
            loss = self.criterion(x_hat,
                                  x.reshape((x.size(0), 784)).to(
                                      self.device)) - self.model.encoder.kl

            loss.backward()
            running_loss += loss.item()

            self.optimizer.step()

        return running_loss / (len(loader) * loader.batch_size)

    def eval(self, loader: DataLoader) -> float:
        predictive_ELBO = 0.0

        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x_hat = self.model(x.reshape((x.size(0), 784)).to(self.device))
                ELBO_batch = -self.criterion(
                    x_hat,
                    x.reshape((x.size(0), 784)).to(
                        self.device)) + self.model.encoder.kl

                predictive_ELBO += ELBO_batch.item()

        return predictive_ELBO / (len(loader) * loader.batch_size)


class VAENoPretrainingNotMNIST(VAEFashionMNIST2notMNISTTrainer):

    def finetune(self):
        train_loader, valid_loader = self.create_finetuning_dataloaders()
        self.model = VAEForClassification(n_latent_dims=2,
                                          n_out_dims=10).to(self.device)

        # model surgery for image classification
        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate,
                                             amsgrad=True)
        self.criterion = nn.CrossEntropyLoss()

        training_loss = []
        val_loss = []
        training_accuracy = []
        val_accuracy = []

        for i in trange(self.finetune_epochs):
            training_loss.append(self.finetune_train(train_loader))
            val_loss.append(self.finetune_train(valid_loader))
            training_accuracy.append(self.finetune_eval(train_loader))
            val_accuracy.append(self.finetune_eval(valid_loader))

            logging.info(
                f'epoch: {i} training loss: {training_loss[-1]:.3f} val loss:{val_loss[-1]:.3f} training accuracy: {training_accuracy[-1]:.3f} val acc: {val_accuracy[-1]:.3f}'
            )

        name = 'vae_no_pretraining_not_mnist'
        #self.save_model(name=name)


class VAEFashionMNIST2MNISTTrainer(VAENotMNIST2MNISTTrainer):

    def create_pretraining_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        transform = transforms.Compose([transforms.ToTensor()])
        MNIST_data_train = torchvision.datasets.FashionMNIST(
            self.data_dir, train=True, transform=transform, download=False)

        train_set, val_set = torch.utils.data.random_split(
            MNIST_data_train, [50000, 10000])
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            val_set, batch_size=len(MNIST_data_train), shuffle=False)
        return train_loader, valid_loader

    def finetune(self):
        train_loader, valid_loader = self.create_finetuning_dataloaders()
        self.model = VAE(n_latent_dims=2).to(self.device)
        self.model.load_state_dict(
            torch.load(
                f'{self.save_dir}models/vae_pretrained_fashion_mnist.pt'))

        # model surgery for image classification
        self.old_decoder = self.model.decoder
        self.model.decoder = nn.Linear(2, 10).to(self.device)
        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate,
                                             amsgrad=True)
        self.criterion = nn.CrossEntropyLoss()

        training_loss = []
        val_loss = []
        training_accuracy = []
        val_accuracy = []

        for i in trange(self.finetune_epochs):
            training_loss.append(self.finetune_train(train_loader))
            val_loss.append(self.finetune_train(valid_loader))
            training_accuracy.append(self.finetune_eval(train_loader))
            val_accuracy.append(self.finetune_eval(valid_loader))

            logging.info(
                f'epoch: {i} training loss: {training_loss[-1]:.3f} val loss:{val_loss[-1]:.3f} training accuracy: {training_accuracy[-1]:.3f} val acc: {val_accuracy[-1]:.3f}'
            )

        name = 'vae_pretrained_notmnist_finetune_mnist'