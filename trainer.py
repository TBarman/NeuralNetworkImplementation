class Trainer:
    def __init__(self, model, data_module, optimizer, epochs=50):
        self.model = model
        self.data_module = data_module
        self.optimizer = optimizer
        self.epochs = epochs

        self.train_dataloader = self.data_module.train_dataloader()
        self.val_dataloader = self.data_module.val_dataloader()


    def fit(self):
        for i in range(self.epochs):
            avg_train_loss, avg_val_loss = self.fit_epoch()
            print(f"Epoch {i+1}: training loss = {avg_train_loss:.4f}, validation loss = {avg_val_loss:.4f}")

    def fit_epoch(self):
        cumulative_train_loss = 0
        cumulative_val_loss = 0
        train_num_batches = len(self.train_dataloader)
        val_num_batches = len(self.val_dataloader)
        for X, Y in self.train_dataloader:
            Z2 = self.model.forward(X)
            J = self.model.loss(Z2, Y)
            cumulative_train_loss += J.item()
            grads = self.model.backward(Y)
            self.optimizer.step(self.model, grads)
        avg_train_loss = cumulative_train_loss/train_num_batches

        for X, Y in self.val_dataloader:
            Z2 = self.model.forward(X)
            J = self.model.loss(Z2, Y)
            cumulative_val_loss += J.item()
        avg_val_loss = cumulative_val_loss/val_num_batches

        return avg_train_loss, avg_val_loss





