class Trainer:
    def __init__(self, model, data_module, optimizer, epochs=50):
        self.model = model
        self.data_module = data_module
        self.optimizer = optimizer
        self.epochs = epochs


    def fit(self):
        for i in range(self.epochs):
            avg_loss = self.fit_epoch()
            print(f"Epoch {i+1}: training loss = {avg_loss}")

    def fit_epoch(self):
        cumulative_loss = 0
        train_dataloader = self.data_module.train_dataloader()
        num_batches = len(train_dataloader)
        for X, Y in train_dataloader:
            Z2 = self.model.forward(X)
            J = self.model.loss(Z2, Y)
            cumulative_loss += J.item()
            grads = self.model.backward(Y)
            self.optimizer.step(self.model, grads)
        return cumulative_loss/num_batches


