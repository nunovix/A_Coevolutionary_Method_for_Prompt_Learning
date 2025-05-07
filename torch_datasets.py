from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, i):
        sample = self.data[i]
        return sample

    def __len__(self):
        return len(self.data)