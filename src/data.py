from torch.utils.data import Dataset


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