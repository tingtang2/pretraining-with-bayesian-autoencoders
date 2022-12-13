from typing import Tuple

import torch
from trainers.svi_optimizer import OptimN2N
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from trainers.vae_trainer import VAENotMNIST2MNISTTrainer

from models.vae import SA_VAE


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
    return nll + kl


class SA_VAENotMNIST2MNISTTrainer(VAENotMNIST2MNISTTrainer):

    def __init__(self, **kwargs) -> None:
        super(VAENotMNIST2MNISTTrainer, self).__init__(**kwargs)
        self.pretrain_name = 'sa_vae_pretrained_notmnist'
        self.finetune_name = 'sa_vae_pretrained_notmnist_finetune_mnist'

        self.model = SA_VAE(n_latent_dims=2,
                            intermediate_size=512,
                            input_size=784).to(self.device)
        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate,
                                             amsgrad=True)

        update_params = list(self.model.decoder.parameters())
        self.meta_optimizer = OptimN2N(loss_fn=variational_loss,
                                       model=self.model,
                                       model_update_params=update_params,
                                       max_grad_norm=5)
        self.max_grad_norm = 5

    def train(self, loader: DataLoader):
        self.model.train()
        running_loss = 0.0
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
            train_nll_vae += nll_vae * self.batch_size
            kl_vae = self.model.encoder.kl
            train_kl_vae += kl_vae

            # forward SVI steps
            var_params = torch.cat([mu, sigma], 1)
            mu_svi = mu
            sigma_svi = sigma

            assert mu_svi.requires_grad == True
            assert sigma_svi.requires_grad == True

            var_params_svi = self.meta_optimizer.forward(
                input=[mu_svi, sigma_svi], y=x)
            mean_svi_final, logvar_svi_final = var_params_svi

            z_samples = self.model.sample(mean_svi_final, logvar_svi_final)
            preds = self.model.decoder(z_samples)

            nll_svi = self.criterion(preds, x)
            train_nll_svi += nll_svi * self.batch_size
            kl_svi = self.model.kl
            train_kl_svi += kl_svi

            # compute loss using VAE + SVI variational param
            var_loss = nll_svi + kl_svi
            var_loss.backward(retain_graph=True)

            # perform backwards through SVI to VAE
            var_param_grads = self.meta_optimizer.backward(
                [mean_svi_final.grad, logvar_svi_final.grad])
            var_param_grads = torch.cat(var_param_grads, 1)
            var_params.backward(var_param_grads)

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_grad_norm)

        return train_nll_vae / (len(loader) * loader.batch_size)

    def eval(self, loader: DataLoader) -> float:
        predictive_ELBO = 0.0

        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x_hat = self.model(x)
                ELBO_batch = -self.criterion(x_hat, x) + self.model.encoder.kl

                predictive_ELBO += ELBO_batch.item()

        return predictive_ELBO / (len(loader) * loader.batch_size)
