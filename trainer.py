import torch
class Trainer:
    def __init__(self, model, data_module, optimizer, epochs=50):
        self.model = model
        self.data_module = data_module
        self.optimizer = optimizer
        self.epochs = epochs

    def fit(self):
        for i in range(self.epochs):
            avg_train_loss, avg_val_loss, avg_train_accuracy, avg_val_accuracy = self.fit_epoch()
            print(f"Epoch {i+1}: training loss = {avg_train_loss:.4f}, validation loss = {avg_val_loss:.4f}, training accuracy = {avg_train_accuracy:.4f}, validation accuracy = {avg_val_accuracy:.4f}")

    def fit_epoch(self):
        train_dataloader = self.data_module.train_dataloader()
        val_dataloader = self.data_module.val_dataloader()

        # Variables to calculate loss
        cumulative_train_loss = 0
        cumulative_val_loss = 0
        train_num_batches = len(train_dataloader)
        val_num_batches = len(val_dataloader)

        # Variables to calculate accuracy
        train_correct = 0
        val_correct = 0
        train_total = 0
        val_total = 0

        # Training loop
        for X, Y in train_dataloader:
            Z2 = self.model.forward(X)
            J = self.model.loss(Z2, Y)
            cumulative_train_loss += J.item()
            grads = self.model.backward(Y)
            self.optimizer.step(self.model, grads)

            train_correct += (torch.argmax(Z2, dim=1) == torch.argmax(Y, dim=1)).sum().item()
            train_total += Y.shape[0]

        avg_train_loss = cumulative_train_loss/train_num_batches
        avg_train_accuracy = train_correct/train_total

        # Validation loop
        for X, Y in val_dataloader:
            Z2 = self.model.forward(X)
            J = self.model.loss(Z2, Y)
            cumulative_val_loss += J.item()

            val_correct += (torch.argmax(Z2, dim=1) == torch.argmax(Y, dim=1)).sum().item()
            val_total += Y.shape[0]

        avg_val_loss = cumulative_val_loss/val_num_batches
        avg_val_accuracy = val_correct/val_total

        return avg_train_loss, avg_val_loss, avg_train_accuracy, avg_val_accuracy





