class Trainer:
    def __init__(self, model, data_module, optimizer, iterations=50):
        self.model = model
        self.data_module = data_module
        self.optimizer = optimizer

        self.iterations = iterations

    def fit(self):
        for i in range(self.iterations):
            self.fit_epoch(self.model, self.data_module)

    def fit_epoch(self):
        for X, Y in self.data_module.train_batches():
            Z2 = self.model.forward(X)
            J = self.model.loss(Z2, Y)
            grads = self.model.backward(Y)
            self.optimizer.step(self.model, grads)
