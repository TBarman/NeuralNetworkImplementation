import torch
class Model:
    '''
    Class representing an MLP with one hidden layer. In general, it is assumed that labels Y are one-hot encoded.
    '''
    def __init__(self, input_size, hidden_size, output_size, decay_weight=0):
        self.decay_weight = decay_weight
        self.W1 = torch.randn((input_size, hidden_size)) * (2/(input_size))**0.5 # He initialization
        self.b1 = torch.zeros((1, hidden_size))
        self.W2 = torch.randn((hidden_size, output_size)) * (2/(hidden_size))**0.5
        self.b2 = torch.zeros((1, output_size))

        self.cache = {}
    def _relu(self, Z):
        mask = Z > 0
        self.cache["mask"] = mask
        return mask * Z

    def _softmax(self, Z):
        c = torch.max(Z, dim=1, keepdims=True).values
        temp = torch.exp(Z - c)
        return temp / torch.sum(temp, dim=1, keepdims=True)

    def _log_softmax(self, Z):
        c = torch.max(Z, dim=1, keepdim=True).values  # subtract max for stability
        log_sum_exp = torch.log(torch.sum(torch.exp(Z - c), dim=1, keepdim=True))
        return (Z - c) - log_sum_exp  # never computes raw probabilities

    def forward(self, X):
        '''
        Returns raw logits of the output layer
        '''
        Z1 = X @ self.W1 + self.b1
        A1 = self._relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        self.cache["X"] = X
        self.cache["Z1"] = Z1
        self.cache["A1"] = A1
        self.cache["Z2"] = Z2
        return Z2

    def loss(self, Z2, Y):
        log_probs = self._log_softmax(Z2)
        L = -torch.trace(Y @ log_probs.T)/Z2.shape[0] # Z2.shape[0] is batch size
        s = (self.decay_weight / 2) * (torch.sum(self.W1**2) + torch.sum(self.W2**2))
        J = L + s
        return J

    def backward(self, Y):
        X, Z1, A1, Z2, mask = self.cache["X"], self.cache["Z1"], self.cache["A1"], self.cache["Z2"], self.cache["mask"]
        batch_size = Y.shape[0]
        dL_dZ2 = (self._softmax(Z2) - Y)/batch_size
        dL_dW2 = A1.T @ dL_dZ2
        dL_db2 = dL_dZ2.sum(dim=0, keepdim=True)

        dL_dZ1 = (mask * (dL_dZ2 @ self.W2.T))
        dL_dW1 = X.T @ dL_dZ1
        dL_db1 = dL_dZ1.sum(dim=0, keepdim=True)

        ds_dW2 = self.decay_weight * self.W2
        ds_db2 = 0
        ds_dW1 = self.decay_weight * self.W1
        ds_db1 = 0

        dJ_dW2 = dL_dW2 + ds_dW2
        dJ_db2 = dL_db2 + ds_db2
        dJ_dW1 = dL_dW1 + ds_dW1
        dJ_db1 = dL_db1 + ds_db1

        return {
            "W2": dJ_dW2,
            "b2": dJ_db2,
            "W1": dJ_dW1,
            "b1": dJ_db1
        }

    def parameters(self):
        return {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2
        }

    def predict(self, X):
        Z2 = self.forward(X)
        return torch.argmax(Z2, dim=1)

