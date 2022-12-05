# base class for active learning experiments
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

import pandas as pd
import plotly.express as px
import torch


class BaseTrainer(ABC):

    def __init__(self,
                 model_type,
                 optimizer_type,
                 acquisition_fn_type,
                 criterion,
                 device: str,
                 save_dir: Union[str, Path],
                 save_plots: bool = True,
                 seed: int = 11202022,
                 **kwargs) -> None:
        super().__init__()

        # basic configs every training run needs
        self.model_type = model_type
        self.optimizer_type = optimizer_type
        self.acquisition_fn_type = acquisition_fn_type
        self.criterion = criterion
        self.device = torch.device(device)
        self.save_plots = save_plots
        self.save_dir = save_dir
        self.seed = seed

        # extra configs in form of kwargs
        for key, item in kwargs.items():
            setattr(self, key, item)

    @abstractmethod
    def create_dataloaders(self):
        pass

    @abstractmethod
    def pre_train(self):
        pass

    @abstractmethod
    def fine_tune(self):
        pass

    @abstractmethod
    def eval(self):
        pass