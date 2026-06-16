from torchvision import datasets, transforms

train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())