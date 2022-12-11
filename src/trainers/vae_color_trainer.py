import logging

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from data import TinyImageNet
from models.vae import VAE, VAEForClassification
from trainers.base_trainer import BaseTrainer


class VAEColorTrainer(BaseTrainer):

    def __init__(self, **kwargs) -> None:
        super(VAEColorTrainer, self).__init__(**kwargs)

        self.model = VAE(n_latent_dims=2,
                         intermediate_size=2048,
                         input_size=3072).to(self.device)
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

    def finetune_train(self, loader: DataLoader):
        self.model.train()
        running_loss = 0.0

        for i, (x, y) in enumerate(loader):
            self.optimizer.zero_grad()

            reshaped_x = x.reshape(x.size(0), self.x_dim)

            y_hat = self.model(reshaped_x.to(self.device))
            # see what happens when we keep self.model.kl (it's really bad!)
            loss = self.criterion(y_hat,
                                  y.to(self.device))  # - self.model.encoder.kl

            loss.backward()
            running_loss += loss.item()

            self.optimizer.step()

        return running_loss / len(loader.dataset)

    def finetune_eval(self, loader: DataLoader):
        num_right = 0
        running_loss = 0.0

        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                reshaped_x = x.reshape(x.size(0), self.x_dim)
                y_hat = self.model(reshaped_x.to(self.device))
                num_right += torch.sum(
                    y.to(self.device) == torch.argmax(
                        y_hat, dim=-1)).detach().cpu().item()

                running_loss += self.criterion(y_hat, y.to(self.device)).item()

        return num_right / len(loader.dataset), (running_loss /
                                                 len(loader.dataset))


class VAENoPretrainingCIFAR10Trainer(VAEColorTrainer):

    def __init__(self, **kwargs) -> None:
        super(VAENoPretrainingCIFAR10Trainer, self).__init__(**kwargs)
        self.finetune_name = 'vae_no_pretraining_cifar10'

    def create_pretraining_dataloaders(self):
        raise NotImplementedError

    def pretrain(self):
        raise NotImplementedError

    def create_finetuning_dataloaders(self):
        transform = transforms.Compose([transforms.ToTensor()])
        CIFAR10_data_train = torchvision.datasets.CIFAR10(self.data_dir,
                                                          train=True,
                                                          transform=transform,
                                                          download=False)

        torch.manual_seed(self.seed)
        train_set, val_set = torch.utils.data.random_split(
            CIFAR10_data_train, [40000, 10000])
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        valid_loader = torch.utils.data.DataLoader(val_set,
                                                   batch_size=len(val_set),
                                                   shuffle=False)
        return train_loader, valid_loader

    def finetune(self):
        train_loader, valid_loader = self.create_finetuning_dataloaders()
        self.model = VAEForClassification(input_size=self.x_dim,
                                          n_latent_dims=2,
                                          n_out_dims=10).to(self.device)

        # model surgery for image classification
        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate,
                                             amsgrad=True)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

        training_loss = []
        val_loss = []
        training_accuracy = []
        val_accuracy = []

        for i in trange(self.finetune_epochs):
            training_loss.append(self.finetune_train(train_loader))
            training_accuracy.append(self.finetune_eval(train_loader)[0])
            acc, loss = self.finetune_eval(valid_loader)
            val_accuracy.append(acc)
            val_loss.append(loss)

            logging.info(
                f'epoch: {i} training loss: {training_loss[-1]:.3f} val loss:{val_loss[-1]:.3f} training accuracy: {training_accuracy[-1]:.3f} val acc: {val_accuracy[-1]:.3f}'
            )


class VAETinyImageNetPreTrainer(VAEColorTrainer):

    def __init__(self, **kwargs) -> None:
        super(VAETinyImageNetPreTrainer, self).__init__(**kwargs)
        self.pretrain_name = 'vae_just_pretrain_tiny_imagenet'

    def create_pretraining_dataloaders(self):
        loaded_tiny_imagenet = TinyImageNet(data_dir=self.data_dir,
                                            num_workers=2)
        return loaded_tiny_imagenet.train_loader, loaded_tiny_imagenet.val_loader

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

        self.save_model(name=self.pretrain_name)
        self.plot_latent(loader=train_loader, name=self.pretrain_name)
        self.plot_reconstructed_color(name=self.pretrain_name)
        self.save_metrics(training_elbo,
                          name=self.pretrain_name + '_training_elbo',
                          phase='pretrain')
        self.save_metrics(predictive_elbo,
                          name=self.pretrain_name + '_predictive_elbo',
                          phase='pretrain')

    def create_finetuning_dataloaders(self):
        raise NotImplementedError

    def finetune(self):
        raise NotImplementedError

    def plot_latent(self, loader, name: str):
        self.model.eval()

        labels = []
        z_s = []
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                if i > 1000:
                    break
                z, mu, sigma = self.model.encoder(
                    x.reshape((x.size(0), self.x_dim)).to(self.device))
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

    def plot_reconstructed_color(self,
                                 r0=(-5, 10),
                                 r1=(-10, 5),
                                 n=12,
                                 name: str = 'default'):
        w = 28
        img = np.zeros((n * w, n * w))
        for i, y in enumerate(np.linspace(*r1, n)):
            for j, x in enumerate(np.linspace(*r0, n)):
                z = torch.Tensor([[x, y]]).to(self.device)
                x_hat = self.model.decoder(z)
                x_hat = x_hat.reshape(w, w, 3).cpu().detach().numpy()

                img[(n - 1 - i) * w:(n - 1 - i + 1) * w,
                    j * w:(j + 1) * w, :] = x_hat

        fig = px.imshow(img,
                        color_continuous_scale=px.colors.sequential.Electric)

        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.write_html(self.save_dir + f'plots/{name}_reconstruction.html')
        fig.write_image(self.save_dir + f'plots/{name}_reconstruction.png')
