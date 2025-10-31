
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_mnist_loaders(data_dir='./data/mnist', batch_size=256, num_workers=4, 
                     sampler=None, distributed=False):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, 
                                   transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, 
                                  transform=transform)

    use_cuda = torch.cuda.is_available()

    if sampler is not None:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=sampler,
            num_workers=num_workers, 
            pin_memory=use_cuda,
            persistent_workers=(num_workers > 0) if distributed else False
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=use_cuda
        )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=use_cuda
    )

    return train_loader, test_loader, train_dataset, test_dataset

def get_vinafood_loaders(train_dir, test_dir, batch_size=128, num_workers=4, 
                        img_size=224, model_type='googlenet', sampler=None,
                        distributed=False):
    if model_type == 'lenet':
        img_size = 32

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    use_cuda = torch.cuda.is_available()

    if sampler is not None:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=sampler,
            num_workers=num_workers, 
            pin_memory=use_cuda,
            persistent_workers=(num_workers > 0) if distributed else False
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=use_cuda
        )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=use_cuda
    )

    return train_loader, test_loader, train_dataset, test_dataset

def get_data_loaders(dataset_type, **kwargs):
    if dataset_type == 'mnist':
        return get_mnist_loaders(**kwargs)
    elif dataset_type == 'vinafood':
        return get_vinafood_loaders(**kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")