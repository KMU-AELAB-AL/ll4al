import torchvision.transforms as T


class Cifar(object):
    def __init__(self):
        self.train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
        ])

        self.test_transform = T.Compose([
            T.ToTensor(),
        ])


class Caltech(object):
    def __init__(self):
        self.train_transform = T.Compose([
            T.Grayscale(),
            T.Resize(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=224, padding=4),
            T.ToTensor(),
        ])

        self.test_transform = T.Compose([
            T.Grayscale(),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
        ])