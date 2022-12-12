import torch
from svi_optimizer import OptimN2N
from torch.utils.data import DataLoader
from vae_trainer import VAENotMNIST2MNISTTrainer

from models.vae import SA_VAE


class SA_VAENotMNIST2MNISTTrainer(VAENotMNIST2MNISTTrainer):

    def __init__(self, **kwargs) -> None:
        super(VAENotMNIST2MNISTTrainer, self).__init__(**kwargs)
        self.pretrain_name = 'vae_pretrained_notmnist'
        self.finetune_name = 'vae_pretrained_notmnist_finetune_mnist'

        self.model = SA_VAE(n_latent_dims=2,
                            intermediate_size=512,
                            input_size=784).to(self.device)
        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate,
                                             amsgrad=True)

        variational_loss = (lambda input, img, model, z=None: z)
        self.meta_optimizer = OptimN2N(variational_loss,
                                       self.model,
                                       update_params,
                                       eps=args.eps,
                                       lr=[args.svi_lr1, args.svi_lr2],
                                       iters=args.svi_steps,
                                       momentum=args.momentum,
                                       acc_param_grads=args.train_n2n == 1,
                                       max_grad_norm=args.svi_max_grad_norm)

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
            var_params_svi = self.meta_optimizer.forward(
                [mean_svi, logvar_svi], img, t % args.print_every == 0)
            mean_svi_final, logvar_svi_final = var_params_svi
            z_samples = model._reparameterize(mean_svi_final, logvar_svi_final)
            preds = model._dec_forward(img, z_samples)
            nll_svi = utils.log_bernoulli_loss(preds, img)
            train_nll_svi += nll_svi.data[0] * batch_size
            kl_svi = utils.kl_loss_diag(mean_svi_final, logvar_svi_final)
            train_kl_svi += kl_svi.data[0] * batch_size
            var_loss = nll_svi + args.beta * kl_svi
            var_loss.backward(retain_graph=True)
            if args.train_n2n == 0:
                if args.train_kl == 1:
                    mean_final = mean_svi_final.detach()
                    logvar_final = logvar_svi_final.detach()
                    kl_init_final = utils.kl_loss(mean, logvar, mean_final,
                                                  logvar_final)
                    kl_init_final.backward(retain_graph=True)
                else:
                    vae_loss = nll_vae + args.beta * kl_vae
                    var_param_grads = torch.autograd.grad(vae_loss,
                                                          [mean, logvar],
                                                          retain_graph=True)
                    var_param_grads = torch.cat(var_param_grads, 1)
                    var_params.backward(var_param_grads, retain_graph=True)
            else:
                var_param_grads = meta_optimizer.backward(
                    [mean_svi_final.grad, logvar_svi_final.grad],
                    t % args.print_every == 0)
                var_param_grads = torch.cat(var_param_grads, 1)
                var_params.backward(var_param_grads)
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(),
                                              args.max_grad_norm)

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
