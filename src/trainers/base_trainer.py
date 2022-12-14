# base class for active learning experiments
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

import pandas as pd
import plotly.express as px
import torch
import numpy as np

from tqdm import tqdm


class BaseTrainer(ABC):

    def __init__(self,
                 optimizer_type,
                 criterion,
                 device: str,
                 save_dir: Union[str, Path],
                 batch_size: int,
                 dropout_prob: float,
                 learning_rate: float,
                 save_plots: bool = True,
                 seed: int = 11202022,
                 **kwargs) -> None:
        super().__init__()

        # basic configs every trainer needs
        self.optimizer_type = optimizer_type
        self.criterion = criterion
        self.device = torch.device(device)
        self.save_plots = save_plots
        self.save_dir = save_dir
        self.seed = seed
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate

        # extra configs in form of kwargs
        for key, item in kwargs.items():
            setattr(self, key, item)

    @abstractmethod
    def create_pretraining_dataloaders(self):
        pass

    @abstractmethod
    def create_finetuning_dataloaders(self):
        pass

    @abstractmethod
    def pretrain(self):
        pass

    @abstractmethod
    def finetune(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    def save_model(self, name: str):
        torch.save(self.model.state_dict(), f'{self.save_dir}models/{name}.pt')

    def plot_latent(self, loader, name: str):
        self.model.eval()

        labels = []
        z_s = []
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(loader)):
                x = x.reshape(x.size(0), -1) #self.x_dim
                x = x.to(self.device)
                z, mu, sigma = self.model.encoder(x)
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
        if self.model_type == "vae_omniglot":
            w = 105
        else:
            w = 28
        print("w: {}".format(w))
        img = np.zeros((n * w, n * w))
        for i, y in enumerate(np.linspace(*r1, n)):
            for j, x in enumerate(np.linspace(*r0, n)):
                z = torch.Tensor([[x, y]]).to(self.device)
                x_hat = self.model.decoder(z)
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

    def save_metrics(self, metrics: List[float], name: str, phase: str):
        save_name = f'{name}_{phase}.json'
        with open(Path(self.save_dir, 'metrics', save_name), 'w') as f:
            json.dump(metrics, f)