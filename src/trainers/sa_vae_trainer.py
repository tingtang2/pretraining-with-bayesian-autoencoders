import logging
from typing import Tuple

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import trange

from models.vae import SA_VAE
from trainers.svi_optimizer import OptimN2N
from trainers.vae_trainer import VAENotMNIST2MNISTTrainer


def variational_loss(input: Tuple,
                     img,
                     model,
                     rand=None,
                     criterion=MSELoss(reduction='mean')):
    mu, sigma = input
    z_samples = model.sample(mu, sigma, rand)
    preds = model.decoder(z_samples)
    nll = criterion(preds, img)
    kl = model.kl
    return nll - kl


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
            max_grad_norm=self.max_grad_norm)

    def train(self, loader: DataLoader):
        self.model.train()
        train_nll_vae = 0.
        train_kl_vae = 0.
        train_nll_svi = 0.
        train_kl_svi = 0.

        for i, (x, y) in enumerate(loader):
            self.optimizer.zero_grad()

            # normal VAE forward
            z, mu, sigma = self.model.encoder(x)
            x_hat = self.model.decoder(z)
            nll_vae = self.criterion(x_hat, x)
            train_nll_vae += nll_vae
            kl_vae = self.model.encoder.kl
            train_kl_vae += kl_vae

            # forward SVI steps
            var_params = torch.cat([mu, sigma], 1)
            mu_svi = mu
            sigma_svi = sigma

            assert mu_svi.requires_grad == True
            assert sigma_svi.requires_grad == True

            var_params_svi = self.meta_optimizer.forward(
                input=[mu_svi, sigma_svi], y=x, verbose=i % 10000 == 0)
            mean_svi_final, logvar_svi_final = var_params_svi

            z_samples = self.model.sample(mean_svi_final, logvar_svi_final)
            preds = self.model.decoder(z_samples)

            nll_svi = self.criterion(preds, x)
            train_nll_svi += nll_svi
            kl_svi = self.model.kl
            train_kl_svi += kl_svi

            # compute loss using VAE + SVI variational param
            var_loss = nll_svi + kl_svi
            var_loss.backward(retain_graph=True)

            # perform backwards through SVI to VAE
            var_param_grads = self.meta_optimizer.backward(
                [mean_svi_final.grad, logvar_svi_final.grad],
                verbose=i % 10000 == 0)
            var_param_grads = torch.cat(var_param_grads, 1)
            var_params.backward(var_param_grads)

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_grad_norm)
            self.optimizer.step()

        return [
            metric / (len(loader) * loader.batch_size) for metric in
            [train_nll_vae, train_kl_vae, train_nll_svi, train_kl_svi]
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
            var_loss = nll_svi + kl_svi

        return [
            metric / (len(loader) * loader.batch_size) for metric in
            [total_nll_vae, total_kl_vae, total_nll_svi, total_kl_svi]
        ]

    def pretrain(self):
        train_loader, valid_loader = self.create_pretraining_dataloaders()

        training_elbo = []
        predictive_elbo = []

        for i in trange(self.pretrain_epochs):
            train_nll_vae, train_kl_vae, train_nll_svi, train_kl_svi = self.train(
                train_loader)
            training_elbo.append(-train_nll_svi.item() + train_kl_svi.item())
            eval_nll_vae, eval_kl_vae, eval_nll_svi, eval_kl_svi = self.eval(
                valid_loader)
            predictive_elbo.append(-eval_nll_svi.item() + eval_kl_svi.item())
            logging.info(
                f'epoch: {i} train_elbo_svi: {training_elbo[-1]:.3f}, train_elbo_vae: {-train_nll_vae.item() + train_kl_vae.item():.3f} predictive_elbo_svi: {predictive_elbo[-1]:.3f}, predictive_elbo_vae: {-eval_nll_vae.item() + eval_kl_vae.item():.3f}'
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
