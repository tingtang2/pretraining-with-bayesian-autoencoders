from base_trainer import BaseTrainer


class VAETrainer(BaseTrainer):

    def __init__(self, **kwargs) -> None:
        super(VAETrainer, self).__init__(**kwargs)

    def pretrain(self):
        pass