from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import os
from torchvision import datasets


class notMNISTDataset(Dataset):
    '''
        torch dataset class for easy batching of notMNIST dataset
    '''

    def __init__(self, img_data, labels):
        self.img_data = img_data
        self.labels = labels

    def __getitem__(self, idx):
        return self.img_data[idx] / 255.0, self.labels[
            idx]  # set images values to be between 0 and 1

    def __len__(self):
        return len(self.labels)


# code adapted from https://github.com/apple/learning-subspaces/blob/master/data/tinyimagenet.py


class TinyImageNet:

    def __init__(self, data_dir: str, batch_size=128, num_workers=1):
        super(TinyImageNet, self).__init__()

        data_root = os.path.join(data_dir, "tiny-imagenet-200")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = ({
            "num_workers": num_workers,
            "pin_memory": True
        } if use_cuda else {})

        # Data loading code
        traindir = os.path.join(data_root, "train")
        valdir = os.path.join(data_root, "val")

        # normalize = transforms.Normalize(
        #     mean=[0.480, 0.448, 0.397], std=[0.276, 0.269, 0.282]
        # )

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                # transforms.RandomCrop(size=64, padding=4),
                # transforms.RandomHorizontalFlip(),
                # transforms.Resize(size=32),
                transforms.ToTensor()
                # normalize,
            ]),
        )

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       **kwargs)

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                # transforms.Resize(size=32),
                transforms.ToTensor()
                # normalize,
            ]),
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=len(val_dataset),
            shuffle=False,
            **kwargs,
        )