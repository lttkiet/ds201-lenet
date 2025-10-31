

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os

from models import ResNet18
from data_utils.loaders import get_vinafood_loaders
from data_utils.training import train_model
from data_utils.evaluation import evaluate_model, print_evaluation
from data_utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    get_distributed_sampler,
    wrap_model_distributed,
    is_main_process,
    get_device
)

def main():
    parser = argparse.ArgumentParser(description='Train ResNet-18 on VinaFood21')
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--test-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-path', type=str, default='./checkpoints/resnet18_vinafood.pt')
    parser.add_argument('--early-stopping-patience', type=int, default=None)
    parser.add_argument('--test-run', action='store_true')
    parser.add_argument('--test-samples', type=int, default=100)
    args = parser.parse_args()

    is_distributed, local_rank, world_size, global_rank = setup_distributed()

    if is_main_process():
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    device = get_device(is_distributed, local_rank)

    if is_main_process():
        print(f'Distributed Training: {is_distributed}')
        if is_distributed:
            print(f'World Size: {world_size}')
        if torch.cuda.is_available():
            print(f'Number of GPUs available: {torch.cuda.device_count()}')
            if is_distributed:
                print(f'Using GPU: {local_rank}')
            elif torch.cuda.device_count() > 1:
                print(f'Using DataParallel with {torch.cuda.device_count()} GPUs')
            else:
                print(f'Using single GPU: cuda:0')
        else:
            print('No GPU available, using CPU')
        print(f'Device: {device}')

    if is_main_process():
        print('Loading VinaFood21 dataset...')

    _, _, train_dataset, test_dataset = get_vinafood_loaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=224,
        model_type='resnet',
        distributed=is_distributed
    )

    train_sampler = get_distributed_sampler(train_dataset, is_distributed, shuffle=True)

    train_loader, test_loader, _, _ = get_vinafood_loaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=224,
        model_type='resnet',
        sampler=train_sampler,
        distributed=is_distributed
    )

    if args.test_run:
        from torch.utils.data import Subset
        
        original_train_dataset = train_dataset
        original_test_dataset = test_dataset
        
        train_indices = list(range(min(args.test_samples, len(train_dataset))))
        train_dataset = Subset(train_dataset, train_indices)
        
        test_indices = list(range(min(args.test_samples // 2, len(test_dataset))))
        test_dataset = Subset(test_dataset, test_indices)
        
        train_sampler = get_distributed_sampler(train_dataset, is_distributed, shuffle=True)
        
        from torch.utils.data import DataLoader
        use_cuda = torch.cuda.is_available()
        
        if train_sampler is not None:
            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                sampler=train_sampler,
                num_workers=args.num_workers, 
                pin_memory=use_cuda,
                persistent_workers=(args.num_workers > 0) if is_distributed else False
            )
        else:
            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=args.num_workers, 
                pin_memory=use_cuda
            )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers, 
            pin_memory=use_cuda
        )
        
        if is_main_process():
            print(f'\n*** TEST RUN MODE ***')
            print(f'Using limited dataset for testing')
        
        num_classes = len(original_train_dataset.classes)
    else:
        num_classes = len(train_dataset.classes)

    if is_main_process():
        print(f'Number of classes: {num_classes}')
        print(f'Training samples: {len(train_dataset)}')
        print(f'Test samples: {len(test_dataset)}')
        if is_distributed:
            print(f'Batch size per GPU: {args.batch_size}')
            print(f'Effective batch size: {args.batch_size * world_size}')

    model = ResNet18(num_classes=num_classes).to(device)

    model = wrap_model_distributed(model, local_rank, is_distributed)

    if is_main_process():
        print(f'\nModel: ResNet-18 with MaxPool between blocks')
        model_params = model.module if hasattr(model, 'module') else model
        print(f'Parameters: {sum(p.numel() for p in model_params.parameters())}')

    criterion = nn.CrossEntropyLoss()

    lr = args.lr * world_size if is_distributed else args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    if is_main_process():
        print(f'\nStarting training for {args.epochs} epochs...')
        print(f'Learning rate: {lr:.6f}')
        if args.early_stopping_patience:
            print(f'Early stopping patience: {args.early_stopping_patience} epochs')

    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        scaler=scaler,
        save_path=args.save_path if is_main_process() else None,
        is_distributed=is_distributed,
        world_size=world_size,
        sampler=train_sampler,
        early_stopping_patience=args.early_stopping_patience
    )

    if is_main_process():
        print('\nFinal Evaluation:')
        val_loss, accuracy, precision, recall, f1 = evaluate_model(
            model, test_loader, criterion, device, is_distributed=is_distributed
        )
        print_evaluation(val_loss, accuracy, precision, recall, f1)

    cleanup_distributed()

if __name__ == '__main__':
    main()
