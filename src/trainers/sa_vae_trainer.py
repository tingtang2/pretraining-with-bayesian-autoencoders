from vae_trainer import VAENotMNIST2MNISTTrainer


class SA_VAENotMNIST2MNISTTrainer(VAENotMNIST2MNISTTrainer):

    def __init__(self, **kwargs) -> None:
        super(VAENotMNIST2MNISTTrainer, self).__init__(**kwargs)
        self.pretrain_name = 'vae_pretrained_notmnist'
        self.finetune_name = 'vae_pretrained_notmnist_finetune_mnist'