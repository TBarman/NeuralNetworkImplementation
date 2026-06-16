from torch.utils.data import DataLoader
class DataModule:
    def __init__(self, train_data, test_data, batch_size):
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size

    def get_dataloader(self, train):
        if train:
            return self.train_dataloader()
        else:
            return self.val_dataloader()

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)