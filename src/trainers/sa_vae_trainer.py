import logging
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from plotly import express as px
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from models.conv_net import ConvNetVAE

from models.vae import SA_VAE
from sa_vae_utils import kl_loss_diag, log_bernoulli_loss
from trainers.svi_optimizer import OptimN2N
from trainers.vae_trainer import VAENotMNIST2MNISTTrainer

from torchvision import transforms
import torchvision


def variational_loss(input, img, model, z=None):
    mean, logvar = input
    z_samples = model.sample(mean, logvar, z)
    preds = model.decoder(z_samples)
    nll = log_bernoulli_loss(preds, img)
    kl = kl_loss_diag(mean, logvar)
    return nll + 1 * kl


def conv_variational_loss(input, img, model, z=None):
    mean, logvar = input
    z_samples = model.reparamatrize(mean, logvar)
    preds = model.decode(z_samples)
    nll = log_bernoulli_loss(preds, img)
    kl = kl_loss_diag(mean, logvar)
    return nll + 1 * kl


# def variational_loss(input: Tuple,
#                      img,
#                      model,
#                      rand=None,
#                      criterion=MSELoss(reduction='sum')):
#     mu, sigma = input
#     z_samples = model.sample(mu, sigma, rand)
#     preds = model.decoder(z_samples)
#     nll = criterion(preds, img)
#     kl = model.kl
#     return nll - kl


class SA_VAENotMNIST2MNISTTrainer(VAENotMNIST2MNISTTrainer):

    def __init__(self, **kwargs) -> None:
        super(VAENotMNIST2MNISTTrainer, self).__init__(**kwargs)
        self.pretrain_name = 'sa_vae_pretrained_notmnist'
        self.finetune_name = 'sa_vae_pretrained_notmnist_finetune_mnist'

        self.model = SA_VAE(n_latent_dims=2,
                            intermediate_size=512,
                            input_size=784).to(self.device)
        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate)

        update_params = list(self.model.decoder.parameters())
        self.meta_optimizer = OptimN2N(
            loss_fn=variational_loss,
            model=self.model,
            lr=[self.svi_learning_rate_mu, self.svi_learning_rate_sigma],
            model_update_params=update_params,
            max_grad_norm=self.max_grad_norm,
            iters=self.num_svi_iterations)

    def train(self, loader: DataLoader):
        self.model.train()
        train_nll_vae = 0.
        train_kl_vae = 0.
        train_nll_svi = 0.
        train_kl_svi = 0.
        total_norm = 0.0

        for i, (x, y) in enumerate(loader):
            self.optimizer.zero_grad()

            # normal VAE forward
            z, mu, sigma = self.model.encoder(x)
            x_hat = self.model.decoder(z)

            reported_nll_vae = self.criterion(x_hat, x)
            train_nll_vae += reported_nll_vae
            reported_kl_vae = self.model.encoder.kl
            train_kl_vae += reported_kl_vae

            # forward SVI steps
            var_params = torch.cat([mu, sigma], 1)
            mu_svi = mu
            sigma_svi = sigma

            assert mu_svi.requires_grad == True
            assert sigma_svi.requires_grad == True

            var_params_svi = self.meta_optimizer.forward(
                input=[mu_svi, sigma_svi], y=x, verbose=i % 10000 == 0)
            mean_svi_final, logvar_svi_final = var_params_svi
            # mean_svi_final.retain_grad()
            # logvar_svi_final.retain_grad()

            z_samples = self.model.sample(mean_svi_final, logvar_svi_final)
            preds = self.model.decoder(z_samples)

            reported_nll_svi = self.criterion(preds, x)
            train_nll_svi += reported_nll_svi
            reported_kl_svi = self.model.kl
            train_kl_svi += reported_kl_svi
            nll_svi = log_bernoulli_loss(preds, x)
            kl_svi = kl_loss_diag(mean_svi_final, logvar_svi_final)

            # compute loss using VAE + SVI variational param
            var_loss = nll_svi + self.beta_svi * kl_svi
            var_loss.backward(retain_graph=True)

            # perform backwards through SVI to VAE
            assert mean_svi_final.grad is not None
            assert logvar_svi_final.grad is not None

            var_param_grads = self.meta_optimizer.backward(
                [mean_svi_final.grad, logvar_svi_final.grad],
                verbose=i % 10000 == 0)
            var_param_grads = torch.cat(var_param_grads, 1)
            var_params.backward(var_param_grads)

            # check norm
            parameters = [
                p for p in self.model.parameters()
                if p.grad is not None and p.requires_grad
            ]
            if len(parameters) == 0:
                total_norm += 0.0
            else:
                device = parameters[0].grad.device
                total_norm += torch.norm(
                    torch.stack([
                        torch.norm(p.grad.detach()).to(device)
                        for p in parameters
                    ]), 2.0).item()

            if self.grad_clip_vae:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.max_grad_norm)
            self.optimizer.step()

        return [
            metric / (len(loader) * loader.batch_size) for metric in [
                train_nll_vae, train_kl_vae, train_nll_svi, train_kl_svi,
                total_norm
            ]
        ]

    def eval(self, loader: DataLoader) -> float:
        total_nll_vae = 0.
        total_kl_vae = 0.
        total_nll_svi = 0.
        total_kl_svi = 0.

        self.model.eval()
        for i, (x, y) in enumerate(loader):
            self.optimizer.zero_grad()

            # normal VAE forward
            z, mu, sigma = self.model.encoder(x)
            x_hat = self.model.decoder(z)
            nll_vae = self.criterion(x_hat, x)
            total_nll_vae += nll_vae
            kl_vae = self.model.encoder.kl
            total_kl_vae += kl_vae

            # forward SVI steps
            mu_svi = mu
            sigma_svi = sigma

            var_params_svi = self.meta_optimizer.forward(
                input=[mu_svi, sigma_svi], y=x)
            mean_svi_final, logvar_svi_final = var_params_svi

            z_samples = self.model.sample(mean_svi_final, logvar_svi_final)
            preds = self.model.decoder(z_samples)

            nll_svi = self.criterion(preds, x)
            total_nll_svi += nll_svi
            kl_svi = self.model.kl
            total_kl_svi += kl_svi

            # compute loss using VAE + SVI variational param
            var_loss = nll_svi - kl_svi

        return [
            metric / (len(loader) * loader.batch_size) for metric in
            [total_nll_vae, total_kl_vae, total_nll_svi, total_kl_svi]
        ]

    def pretrain(self):
        train_loader, valid_loader = self.create_pretraining_dataloaders()

        # # TODO: REPLACE THIS
        # self.model.load_state_dict(
        #     torch.load(
        #         f'{self.save_dir}models/sa_vae_pretrained_notmnist-2022-12-15 01:40:50,217.pt'
        #     ))

        training_elbo = []
        predictive_elbo = []
        predictive_elbo_reconstruction_loss = []

        for i in trange(self.pretrain_epochs):
            train_nll_vae, train_kl_vae, train_nll_svi, train_kl_svi, total_norm = self.train(
                train_loader)
            training_elbo.append(-train_nll_svi.item() + train_kl_svi.item())
            eval_nll_vae, eval_kl_vae, eval_nll_svi, eval_kl_svi = self.eval(
                valid_loader)
            predictive_elbo.append(-eval_nll_svi.item() + eval_kl_svi.item())
            predictive_elbo_reconstruction_loss.append(-eval_nll_svi.item())
            if self.pretrain_name == 'svi_vae_pretrained_notmnist':
                logging.info(
                    f'epoch: {i} train_elbo_svi: {training_elbo[-1]:.3f}, predictive_elbo_svi: {predictive_elbo[-1]:.3f}, total_norm: {total_norm:.3f}, kl train: {train_kl_svi:.3f}, kl eval: {eval_kl_svi:.3f}, reconstruction_loss eval: {predictive_elbo_reconstruction_loss[-1]:.3f}'
                )
            else:
                logging.info(
                    f'epoch: {i} train_elbo_svi: {training_elbo[-1]:.3f}, train_elbo_vae: {-train_nll_vae.item() + train_kl_vae.item():.3f} predictive_elbo_svi: {predictive_elbo[-1]:.3f}, predictive_elbo_vae: {-eval_nll_vae.item() + eval_kl_vae.item():.3f} total_norm: {total_norm:.3f}, kl train: {train_kl_svi:.3f}, kl eval: {eval_kl_svi:.3f}'
                )

        self.save_model(name=self.pretrain_name)
        self.plot_latent(loader=train_loader, name=self.pretrain_name)
        self.plot_reconstructed(name=self.pretrain_name)
        self.save_metrics(training_elbo,
                          name=self.pretrain_name + '_training_elbo',
                          phase='pretrain')
        self.save_metrics(predictive_elbo,
                          name=self.pretrain_name + '_predictive_elbo',
                          phase='pretrain')
        self.save_metrics(predictive_elbo_reconstruction_loss,
                          name=self.pretrain_name +
                          '_predictive_elbo_reconstruction_loss',
                          phase='pretrain')


class SVI_VAENotMNIST2MNISTTrainer(SA_VAENotMNIST2MNISTTrainer):

    def __init__(self, **kwargs) -> None:
        super(SVI_VAENotMNIST2MNISTTrainer, self).__init__(**kwargs)
        self.pretrain_name = 'svi_vae_pretrained_notmnist'
        self.finetune_name = 'svi_vae_pretrained_notmnist_finetune_mnist'

        update_params = list(self.model.decoder.parameters())
        self.meta_optimizer = OptimN2N(
            loss_fn=variational_loss,
            model=self.model,
            lr=[self.svi_learning_rate_mu, self.svi_learning_rate_sigma],
            model_update_params=update_params,
            max_grad_norm=self.max_grad_norm,
            acc_param_grads=False,
            iters=self.num_svi_iterations)
        self.warmup = 10

    def train(self, loader: DataLoader):
        self.model.train()
        train_nll_vae = 0.
        train_kl_vae = 0.
        train_nll_svi = 0.
        train_kl_svi = 0.
        total_norm = 0.0

        for i, (x, y) in enumerate(loader):
            self.optimizer.zero_grad()

            # if self.warmup > 0:
            #     self.beta_svi = min(
            #         1, self.beta_svi + 1. / (self.warmup * len(loader)))

            mu_svi = torch.empty(x.size(0), 2, requires_grad=True).cuda()
            sigma_svi = torch.empty(x.size(0), 2, requires_grad=True).cuda()

            torch.nn.init.kaiming_normal_(mu_svi)
            torch.nn.init.uniform_(sigma_svi, 0, 1)

            assert mu_svi.requires_grad == True
            assert sigma_svi.requires_grad == True

            # x = torch.bernoulli(x)

            # forward SVI steps
            var_params_svi = self.meta_optimizer.forward(
                input=[mu_svi, sigma_svi], y=x, verbose=i % 10000 == 0)
            mean_svi_final, logvar_svi_final = var_params_svi

            z_samples = self.model.sample(mean_svi_final, logvar_svi_final)
            preds = self.model.decoder(z_samples)

            reported_nll_svi = self.criterion(preds, x)
            nll_svi = log_bernoulli_loss(preds, x)
            kl_svi = kl_loss_diag(mean_svi_final, logvar_svi_final)
            train_nll_svi += reported_nll_svi
            reported_kl_svi = self.model.kl
            train_kl_svi += reported_kl_svi

            # compute loss using VAE + SVI variational param
            var_loss = nll_svi + self.beta_svi * kl_svi
            var_loss.backward()

            # check norm
            parameters = [
                p for p in self.model.parameters()
                if p.grad is not None and p.requires_grad
            ]
            if len(parameters) == 0:
                total_norm += 0.0
            else:
                device = parameters[0].grad.device
                total_norm += torch.norm(
                    torch.stack([
                        torch.norm(p.grad.detach()).to(device)
                        for p in parameters
                    ]), 2.0).item()

            self.optimizer.step()

        return [
            metric / (len(loader) * loader.batch_size) for metric in [
                train_nll_vae, train_kl_vae, train_nll_svi, train_kl_svi,
                total_norm
            ]
        ]

    def eval(self, loader: DataLoader) -> float:
        total_nll_vae = 0.
        total_kl_vae = 0.
        total_nll_svi = 0.
        total_kl_svi = 0.

        self.model.eval()
        for i, (x, y) in enumerate(loader):

            mu_svi = torch.empty(x.size(0), 2, requires_grad=True).cuda()
            sigma_svi = torch.empty(x.size(0), 2, requires_grad=True).cuda()

            torch.nn.init.kaiming_normal_(mu_svi)
            torch.nn.init.uniform_(sigma_svi, 0, 1)

            assert mu_svi.requires_grad == True
            assert sigma_svi.requires_grad == True

            # forward SVI steps
            var_params_svi = self.meta_optimizer.forward(
                input=[mu_svi, sigma_svi], y=x, verbose=i % 10000 == 0)
            mean_svi_final, logvar_svi_final = var_params_svi

            z_samples = self.model.sample(mean_svi_final, logvar_svi_final)
            preds = self.model.decoder(z_samples)

            nll_svi = self.criterion(preds, x)
            total_nll_svi += nll_svi
            kl_svi = self.model.kl
            total_kl_svi += kl_svi

            # compute loss using VAE + SVI variational param
            var_loss = nll_svi - kl_svi

        return [
            metric / (len(loader) * loader.batch_size) for metric in
            [total_nll_vae, total_kl_vae, total_nll_svi, total_kl_svi]
        ]

    def plot_latent(self, loader, name: str):
        self.model.eval()

        labels = []
        z_s = []
        for i, (x, y) in enumerate(tqdm(loader)):
            if i > 100:
                break

            x = x.reshape(x.size(0), -1)  #self.x_dim
            x = x.to(self.device)
            mu_svi = torch.empty(x.size(0), 2, requires_grad=True).cuda()
            sigma_svi = torch.empty(x.size(0), 2, requires_grad=True).cuda()

            torch.nn.init.kaiming_normal_(mu_svi)
            torch.nn.init.uniform_(sigma_svi, 0, 1)

            # forward SVI steps
            var_params_svi = self.meta_optimizer.forward(
                input=[mu_svi, sigma_svi], y=x, verbose=i % 10000 == 0)
            mean_svi_final, logvar_svi_final = var_params_svi

            z_samples = self.model.sample(mean_svi_final, logvar_svi_final)
            z_s.append(z_samples.cpu().detach().numpy())

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


class SA_VAEOmniglotTrainer(SA_VAENotMNIST2MNISTTrainer):

    def __init__(self, **kwargs) -> None:
        super(SA_VAEOmniglotTrainer, self).__init__(**kwargs)

        self.model = ConvNetVAE(latent_dim=self.latent_dim_size,
                                bayesian_encoder=False,
                                bayesian_decoder=False).to(self.device)
        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate,
                                             amsgrad=True)
        self.pretrain_name = 'convnet_sa_vae_pretrain_omniglot'

        update_params = list(self.model.decoder.parameters())
        self.meta_optimizer = OptimN2N(
            loss_fn=conv_variational_loss,
            model=self.model,
            lr=[self.svi_learning_rate_mu, self.svi_learning_rate_sigma],
            model_update_params=update_params,
            max_grad_norm=self.max_grad_norm,
            iters=self.num_svi_iterations)

    def create_pretraining_dataloaders(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize(size=104)])
        omniglot_train = torchvision.datasets.Omniglot(root=self.data_dir,
                                                       background=True,
                                                       transform=transform,
                                                       download=False)
        omniglot_val = torchvision.datasets.Omniglot(root=self.data_dir,
                                                     background=False,
                                                     transform=transform,
                                                     download=False)
        torch.manual_seed(self.seed)
        train_subset, _ = torch.utils.data.random_split(
            omniglot_train, [4000, 15280])
        val_subset, _ = torch.utils.data.random_split(omniglot_val,
                                                      [500, 12680])

        train_loader = torch.utils.data.DataLoader(train_subset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        valid_loader = torch.utils.data.DataLoader(val_subset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        print("omniglot data shape: {}".format(
            omniglot_train.__getitem__(0)[0].shape))

        return train_loader, valid_loader

    def train(self, loader: DataLoader):
        self.model.train()
        train_nll_vae = 0.
        train_kl_vae = 0.
        train_nll_svi = 0.
        train_kl_svi = 0.
        total_norm = 0.0

        for i, (x, y) in enumerate(loader):
            self.optimizer.zero_grad()
            x = x.to(self.device)

            if i % 10 == 1:
                print(i)

            # normal VAE forward
            mean, logvar = self.model.encode(x)
            logvar = torch.nan_to_num(logvar, nan=.01, neginf=.01)

            std = torch.exp(logvar / 2) + self.model.eps
            assert torch.all(std > 0)
            z = self.model.reparamatrize(mean, std)
            outputs = self.model.decode(z)
            elbo, reported_nll_vae = self.model.new_loss_fn(
                z, x, outputs, mean, std)

            train_nll_vae += reported_nll_vae
            train_kl_vae += elbo - reported_nll_vae

            # forward SVI steps
            var_params = torch.cat([mean, std], 1)
            mu_svi = mean
            sigma_svi = std

            assert mu_svi.requires_grad == True
            assert sigma_svi.requires_grad == True

            var_params_svi = self.meta_optimizer.forward(
                input=[mu_svi, sigma_svi], y=x, verbose=i % 10000 == 0)
            mean_svi_final, logvar_svi_final = var_params_svi
            # mean_svi_final.retain_grad()
            # logvar_svi_final.retain_grad()

            z_samples = self.model.reparamatrize(mean_svi_final,
                                                 logvar_svi_final)
            preds = self.model.decode(z_samples)
            elbo_svi, reported_nll_svi = self.model.new_loss_fn(
                z, x, outputs, mean, std)

            train_nll_svi += reported_nll_svi
            train_kl_svi += elbo_svi - reported_nll_svi
            nll_svi = log_bernoulli_loss(preds, x)
            kl_svi = kl_loss_diag(mean_svi_final, logvar_svi_final)

            # compute loss using VAE + SVI variational param
            var_loss = nll_svi + self.beta_svi * kl_svi
            var_loss.backward(retain_graph=True)

            # perform backwards through SVI to VAE
            assert mean_svi_final.grad is not None
            assert logvar_svi_final.grad is not None

            var_param_grads = self.meta_optimizer.backward(
                [mean_svi_final.grad, logvar_svi_final.grad],
                verbose=i % 10000 == 0)
            var_param_grads = torch.cat(var_param_grads, 1)
            var_params.backward(var_param_grads)

            # # check norm
            # parameters = [
            #     p for p in self.model.parameters()
            #     if p.grad is not None and p.requires_grad
            # ]
            # if len(parameters) == 0:
            #     total_norm += 0.0
            # else:
            #     device = parameters[0].grad.device
            #     total_norm += torch.norm(
            #         torch.stack([
            #             torch.norm(p.grad.detach()).to(device)
            #             for p in parameters
            #         ]), 2.0).item()

            # if self.grad_clip_vae:
            #     if self.max_grad_norm > 0:
            #         torch.nn.utils.clip_grad_norm_(self.model.parameters(),
            #                                        self.max_grad_norm)
            self.optimizer.step()

        return [
            metric / (len(loader) * loader.batch_size) for metric in [
                train_nll_vae, train_kl_vae, train_nll_svi, train_kl_svi,
                total_norm
            ]
        ]

    def eval(self, loader: DataLoader) -> float:
        total_nll_vae = 0.
        total_kl_vae = 0.
        total_nll_svi = 0.
        total_kl_svi = 0.

        self.model.eval()
        for i, (x, y) in enumerate(loader):
            self.optimizer.zero_grad()
            x = x.to(self.device)

            # normal VAE forward
            mean, logvar = self.model.encode(x)
            logvar = torch.nan_to_num(logvar, nan=.01, neginf=.01)

            std = torch.exp(logvar / 2) + self.model.eps
            assert torch.all(std > 0)
            z = self.model.reparamatrize(mean, std)
            outputs = self.model.decode(z)

            elbo, nll_vae = self.model.new_loss_fn(z, x, outputs, mean, std)
            total_nll_vae += nll_vae
            total_kl_vae += elbo - nll_vae

            # forward SVI steps
            mu_svi = mean
            sigma_svi = std

            var_params_svi = self.meta_optimizer.forward(
                input=[mu_svi, sigma_svi], y=x)
            mean_svi_final, logvar_svi_final = var_params_svi

            z_samples = self.model.reparamatrize(mean_svi_final,
                                                 logvar_svi_final)
            preds = self.model.decode(z_samples)

            elbo_svi, nll_svi = self.model.new_loss_fn(z, x, outputs, mean,
                                                       std)

            total_nll_svi += nll_svi
            total_kl_svi += elbo_svi - nll_svi

            # compute loss using VAE + SVI variational param
            # var_loss = nll_svi - kl_svi

        return [
            metric / (len(loader) * loader.batch_size) for metric in
            [total_nll_vae, total_kl_vae, total_nll_svi, total_kl_svi]
        ]

    def pretrain(self):
        train_loader, valid_loader = self.create_pretraining_dataloaders()

        # # TODO: REPLACE THIS
        # self.model.load_state_dict(
        #     torch.load(
        #         f'{self.save_dir}models/sa_vae_pretrained_notmnist-2022-12-15 01:40:50,217.pt'
        #     ))

        training_elbo = []
        predictive_elbo = []
        predictive_elbo_reconstruction_loss = []

        for i in trange(self.pretrain_epochs):
            train_nll_vae, train_kl_vae, train_nll_svi, train_kl_svi, total_norm = self.train(
                train_loader)
            training_elbo.append(-train_nll_svi.item() + train_kl_svi.item())
            torch.cuda.empty_cache()
            eval_nll_vae, eval_kl_vae, eval_nll_svi, eval_kl_svi = self.eval(
                valid_loader)
            predictive_elbo.append(-eval_nll_svi.item() + eval_kl_svi.item())
            predictive_elbo_reconstruction_loss.append(-eval_nll_svi.item())
            if self.pretrain_name == 'svi_vae_pretrained_notmnist':
                logging.info(
                    f'epoch: {i} train_elbo_svi: {training_elbo[-1]:.3f}, predictive_elbo_svi: {predictive_elbo[-1]:.3f}, total_norm: {total_norm:.3f}, kl train: {train_kl_svi:.3f}, kl eval: {eval_kl_svi:.3f}, reconstruction_loss eval: {predictive_elbo_reconstruction_loss[-1]:.3f}'
                )
            else:
                logging.info(
                    f'epoch: {i} train_elbo_svi: {training_elbo[-1]:.3f}, train_elbo_vae: {-train_nll_vae.item() + train_kl_vae.item():.3f} predictive_elbo_svi: {predictive_elbo[-1]:.3f}, predictive_elbo_vae: {-eval_nll_vae.item() + eval_kl_vae.item():.3f} total_norm: {total_norm:.3f}, kl train: {train_kl_svi:.3f}, kl eval: {eval_kl_svi:.3f}'
                )

        self.save_model(name=self.pretrain_name)
        self.plot_latent(loader=train_loader, name=self.pretrain_name)
        self.plot_reconstructed(name=self.pretrain_name)
        self.save_metrics(training_elbo,
                          name=self.pretrain_name + '_training_elbo',
                          phase='pretrain')
        self.save_metrics(predictive_elbo,
                          name=self.pretrain_name + '_predictive_elbo',
                          phase='pretrain')
        self.save_metrics(predictive_elbo_reconstruction_loss,
                          name=self.pretrain_name +
                          '_predictive_elbo_reconstruction_loss',
                          phase='pretrain')

    def plot_latent(self, loader, name: str):
        self.model.eval()

        labels = []
        z_s = []
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(loader)):
                mean, logvar = self.model.encode(x.to(self.device))
                std = torch.exp(logvar / 2)
                z = self.model.reparamatrize(mean, std)
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

    def plot_reconstructed(self,
                           r0=(-5, 10),
                           r1=(-10, 5),
                           n=12,
                           name: str = 'default'):
        w = 104
        print("w: {}".format(w))
        img = np.zeros((n * w, n * w))
        for i, y in enumerate(np.linspace(*r1, n)):
            for j, x in enumerate(np.linspace(*r0, n)):
                z = torch.Tensor([[x, y]]).to(self.device)
                x_hat = self.model.decode(z)
                x_hat = x_hat.reshape(w, w).cpu().detach().numpy()

                img[(n - 1 - i) * w:(n - 1 - i + 1) * w,
                    j * w:(j + 1) * w] = x_hat

        fig = px.imshow(img,
                        color_continuous_scale=px.colors.sequential.Electric)

        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.write_html(self.save_dir + f'plots/{name}_reconstruction.html')
        fig.write_image(self.save_dir + f'plots/{name}_reconstruction.png')