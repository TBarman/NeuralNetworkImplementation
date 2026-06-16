from torch.utils.data import DataLoader
class DataModule:
    def __init__(self, train_data, val_data, test_data, batch_size):
        self.batch_size = batch_size
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)