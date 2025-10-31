
import torch
import torch.nn as nn
import argparse
import os

from models import LeNet, GoogLeNet, ResNet18, PretrainedResNet50
from data_utils.loaders import get_mnist_loaders, get_vinafood_loaders
from data_utils.evaluation import evaluate_model, print_evaluation

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--model-type', type=str, required=True,
                       choices=['lenet', 'googlenet', 'resnet18', 'pretrained_resnet50'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--train-dir', type=str)
    parser.add_argument('--test-dir', type=str)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if args.model_type == 'lenet':
        if not args.data_dir:
            args.data_dir = './data/mnist'
        print('Loading MNIST dataset...')
        _, test_loader = get_mnist_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        num_classes = 10
    else:
        if not args.test_dir:
            raise ValueError('--test-dir is required for VinaFood21 models')
        print('Loading VinaFood21 dataset...')

        model_type_map = {
            'googlenet': 'googlenet',
            'resnet18': 'resnet',
            'pretrained_resnet50': 'pretrained'
        }

        _, test_loader = get_vinafood_loaders(
            train_dir=args.train_dir or args.test_dir,
            test_dir=args.test_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=224,
            model_type=model_type_map[args.model_type]
        )
        num_classes = len(test_loader.dataset.classes)

    print(f'Test samples: {len(test_loader.dataset)}')
    print(f'Number of classes: {num_classes}')

    print(f'\nLoading {args.model_type} model...')
    if args.model_type == 'lenet':
        model = LeNet(num_classes=num_classes)
    elif args.model_type == 'googlenet':
        model = GoogLeNet(num_classes=num_classes, aux_logits=False)
    elif args.model_type == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    elif args.model_type == 'pretrained_resnet50':
        model = PretrainedResNet50(num_classes=num_classes)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f'Loaded checkpoint from epoch {checkpoint.get("epoch", "unknown")}')
    if 'f1' in checkpoint:
        print(f'Checkpoint F1 score: {checkpoint["f1"]:.4f}')

    criterion = nn.CrossEntropyLoss()
    print('\nEvaluating model...')
    val_loss, accuracy, precision, recall, f1 = evaluate_model(
        model, test_loader, criterion, device
    )
    print_evaluation(val_loss, accuracy, precision, recall, f1)

if __name__ == '__main__':
    main()