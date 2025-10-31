# DS201 LeNet - Deep Learning Models Training

This repository contains implementations of various deep learning models (LeNet, ResNet-18, GoogLeNet, and Pretrained ResNet-50) for image classification tasks on MNIST and VinaFood21 datasets.

## Requirements

### Dependencies

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

Required packages:
- torch>=2.1.0
- torchvision>=0.16.0
- transformers>=4.35.0
- scikit-learn>=1.3.0
- numpy>=1.24.0
- Pillow>=10.0.0
- tqdm>=4.66.0

## CPU Training Instructions

All training scripts support CPU training by default. PyTorch will automatically use the CPU if CUDA is not available.

### Training LeNet on MNIST (CPU)

LeNet is a lightweight model perfect for CPU training on the MNIST dataset.

#### Quick Test Run (Fast training for testing)

```bash
python train_lenet.py --test-run --test-samples 100 --epochs 5
```

This will:
- Use only 100 training samples and 50 test samples
- Train for 5 epochs
- Complete in a few minutes on CPU

#### Full Training on CPU

```bash
python train_lenet.py --epochs 40 --batch-size 256 --lr 0.001
```

Training parameters:
- `--data-dir`: Directory for MNIST data (default: `./data/mnist`)
- `--batch-size`: Batch size for training (default: 256)
- `--epochs`: Number of training epochs (default: 40)
- `--lr`: Learning rate (default: 0.001)
- `--num-workers`: Number of data loading workers (default: 4)
- `--save-path`: Path to save the trained model (default: `./checkpoints/lenet_mnist.pt`)
- `--early-stopping-patience`: Early stopping patience in epochs (optional)
- `--test-run`: Enable test run mode with limited samples
- `--test-samples`: Number of samples for test run (default: 100)

#### Recommended CPU Settings

For optimal CPU performance:

```bash
python train_lenet.py \
  --epochs 40 \
  --batch-size 128 \
  --lr 0.001 \
  --num-workers 2
```

Reducing `batch-size` and `num-workers` can help on systems with limited RAM or CPU cores.

### Training Other Models on CPU

#### ResNet-18 on VinaFood21

```bash
python train_resnet18.py \
  --train-dir ./path/to/train \
  --test-dir ./path/to/test \
  --epochs 30 \
  --batch-size 32 \
  --num-workers 2
```

#### GoogLeNet on VinaFood21

```bash
python train_googlenet.py \
  --train-dir ./path/to/train \
  --test-dir ./path/to/test \
  --epochs 30 \
  --batch-size 32 \
  --num-workers 2
```

#### Pretrained ResNet-50 on VinaFood21

```bash
python train_pretrained_resnet50.py \
  --train-dir ./path/to/train \
  --test-dir ./path/to/test \
  --epochs 20 \
  --batch-size 16 \
  --num-workers 2
```

**Note:** For larger models like ResNet-18, GoogLeNet, and ResNet-50, CPU training will be significantly slower. Consider using smaller batch sizes (16-32) and reducing the number of workers (2-4) for CPU training.

## Dataset Preparation

### MNIST Dataset

The MNIST dataset will be automatically downloaded when you run the training script for the first time. It will be saved to the directory specified by `--data-dir` (default: `./data/mnist`).

### VinaFood21 Dataset

For VinaFood21 dataset, you need to organize your data in the following structure:

```
VinaFood21/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── ...
└── test/
    ├── class1/
    │   └── ...
    └── ...
```

Then use `--train-dir` and `--test-dir` arguments to specify the paths.

## Model Evaluation

To evaluate a trained model:

```bash
python evaluate.py \
  --model-type lenet \
  --checkpoint ./checkpoints/lenet_mnist.pt \
  --data-dir ./data/mnist
```

For other models on VinaFood21:

```bash
python evaluate.py \
  --model-type resnet18 \
  --checkpoint ./checkpoints/resnet18_vinafood.pt \
  --test-dir ./path/to/test
```

Available model types:
- `lenet`: LeNet for MNIST
- `googlenet`: GoogLeNet for VinaFood21
- `resnet18`: ResNet-18 for VinaFood21
- `pretrained_resnet50`: Pretrained ResNet-50 for VinaFood21

## GPU Training

If you have CUDA-enabled GPU(s), the training scripts will automatically detect and use them. No changes to the commands are needed.

### Single GPU

The scripts will automatically use a single GPU if available:

```bash
python train_lenet.py --epochs 40
```

### Multi-GPU Training (Distributed)

For distributed training across multiple GPUs, use `torchrun`:

```bash
torchrun --nproc_per_node=2 train_lenet.py --epochs 40 --batch-size 256
```

Replace `2` with the number of GPUs you want to use.

## Performance Tips for CPU Training

1. **Reduce batch size**: Smaller batches (32-128) use less memory and can be faster on CPU
2. **Reduce workers**: Use 0-4 workers for data loading on CPU
3. **Use test run mode**: Test your setup quickly with `--test-run` flag
4. **Monitor system resources**: Watch CPU and RAM usage to optimize settings
5. **Consider model size**: LeNet is much faster than ResNet-50 on CPU

## Output

Trained models are saved to the `./checkpoints/` directory by default. The training script will:
- Display training progress for each epoch
- Show loss, accuracy, precision, recall, and F1-score
- Save the best model based on validation accuracy
- Perform final evaluation after training completes

## Troubleshooting

### Out of Memory
- Reduce `--batch-size`
- Reduce `--num-workers`
- Close other applications

### Slow Training
- Use `--test-run` to verify setup first
- Reduce model complexity (use LeNet instead of ResNet)
- Consider using a GPU if available

### Dataset Not Found
- Check that paths to `--data-dir`, `--train-dir`, or `--test-dir` are correct
- For MNIST, ensure internet connection for automatic download
- For VinaFood21, verify the directory structure

## Repository Structure

```
.
├── models/
│   ├── lenet.py              # LeNet architecture
│   ├── resnet.py             # ResNet-18 architecture
│   ├── googlenet.py          # GoogLeNet architecture
│   └── pretrained_resnet50.py # Pretrained ResNet-50
├── data_utils/
│   ├── loaders.py            # Data loading utilities
│   ├── training.py           # Training loop
│   ├── evaluation.py         # Evaluation utilities
│   └── distributed.py        # Distributed training utilities
├── train_lenet.py            # Train LeNet on MNIST
├── train_resnet18.py         # Train ResNet-18 on VinaFood21
├── train_googlenet.py        # Train GoogLeNet on VinaFood21
├── train_pretrained_resnet50.py # Train Pretrained ResNet-50
├── evaluate.py               # Model evaluation script
└── requirements.txt          # Python dependencies
```

## License

This project is for educational purposes as part of the DS201 course.
