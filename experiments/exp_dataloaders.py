import torch
from torchvision import datasets, transforms

def getDataLoader(path2data,dataset_name,train_batch=32,test_batch=1024,num_workers=1):
    if dataset_name == 'mnist':
        f_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
        dset = datasets.MNIST

    elif dataset_name == 'cifar10':
        f_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dset = datasets.CIFAR10

    else:
        raise ValueError('{} given: is not a valid option'.format(dataset_name))

    train_loader = torch.utils.data.DataLoader(
        dset(path2data, train=True, download=True,
                       transform=f_transform),
        batch_size=train_batch,
        shuffle=True,
        num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        dset(path2data, train=False, transform=f_transform),
        batch_size=test_batch,
        shuffle=True,
        num_workers=num_workers)

    return train_loader,test_loader
