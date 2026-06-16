from torchvision import datasets, transforms
import torch
from torch.utils.data import TensorDataset
from datamodule import DataModule
from model import Model
from optimizer import Optimizer
from trainer import Trainer
train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

X_train = train.data.float() / 255.0
Y_train = train.targets

X_test = test.data.float() / 255.0
Y_test = test.targets

def one_hot(labels, num_classes=10):
    Y = torch.zeros(len(labels), num_classes)
    Y[torch.arange(len(labels)), labels] = 1
    return Y

X_train = X_train.reshape(X_train.shape[0], -1)
Y_train = one_hot(Y_train)

X_test = X_test.reshape(X_test.shape[0], -1)
Y_test = one_hot(Y_test)

train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)

data_module = DataModule(train_data=train_dataset, test_data=test_dataset, batch_size=32)
model = Model(input_size=X_train.shape[1], hidden_size=256, output_size=Y_train.shape[1])
optimizer = Optimizer()
trainer = Trainer(model, data_module, optimizer)
trainer.fit()