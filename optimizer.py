class Optimizer:
    def __init__(self, learning_rate=0.003):
        self.learning_rate = learning_rate

    def step(self, model, grads):
        parameters = model.parameters()
        for parameter in parameters:
            parameters[parameter] -= self.learning_rate * grads[parameter]
