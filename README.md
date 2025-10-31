# DS201 - Deep Learning Model Training

This repository contains implementations of 4 deep learning models for image classification tasks as part of the DS201 course assignments.

**✨ Flexible GPU/CPU Training Support**
- Automatically detects and uses all available GPUs
- Falls back to CPU when no GPU is available
- Supports single GPU, multiple GPU (DataParallel), and distributed multi-GPU (DDP) training

## Models

1. **LeNet** - Trained on MNIST dataset
2. **GoogLeNet (Inception v1)** - Trained on VinaFood21 dataset
3. **ResNet-18** - Trained on VinaFood21 dataset  
4. **Pretrained ResNet50** - Fine-tuned on VinaFood21 dataset using HuggingFace

## Requirements

- Python 3.12+
- PyTorch 2.1+
- torchvision 0.16+
- transformers 4.35+
- scikit-learn 1.3+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

### MNIST Dataset
The MNIST dataset will be automatically downloaded when running the LeNet training script.

### VinaFood21 Dataset
Download the VinaFood21 dataset from: https://drive.google.com/file/d/1UpZOf0XlwvB4rKpyZ35iwTA8oWHqDBbR/view?usp=share_link

Extract the dataset and note the paths to the `train` and `test` directories.

## Usage

### Training Modes

#### CPU Training
All training scripts automatically detect available hardware. To force CPU-only training (even if GPUs are available), use the `CUDA_VISIBLE_DEVICES=""` environment variable:

```bash
CUDA_VISIBLE_DEVICES="" python train_lenet.py --data-dir ./data/mnist --epochs 40
```

This is useful for:
- Testing on machines without GPUs
- Debugging CPU-specific issues
- Running on CPU-only servers or containers

#### Test Run Mode
For quick testing and debugging, use the `--test-run` flag to train on a limited subset of the data:

```bash
python train_lenet.py --data-dir ./data/mnist --test-run --test-samples 100 --epochs 2
```

The `--test-run` flag:
- Limits training dataset to `--test-samples` samples (default: 100)
- Limits test dataset to half of `--test-samples` samples
- Useful for quick validation of code changes
- Recommended to use with reduced epochs (e.g., `--epochs 2`)
- Can be combined with CPU training for fast local testing

### Automatic Device Detection

All training scripts automatically detect and use available hardware:
- **No GPU**: Trains on CPU
- **1 GPU**: Uses single GPU
- **Multiple GPUs**: Uses DataParallel for efficient multi-GPU training
- **With torchrun**: Uses DistributedDataParallel (DDP) for maximum performance

### Single GPU or CPU Training

All training scripts work out of the box without any special configuration:

#### Bài 1: Train LeNet on MNIST

```bash
python train_lenet.py \
    --data-dir ./data/mnist \
    --batch-size 256 \
    --epochs 40 \
    --lr 0.001
```

**CPU-only training (force CPU even if GPU available):**
```bash
CUDA_VISIBLE_DEVICES="" python train_lenet.py \
    --data-dir ./data/mnist \
    --batch-size 256 \
    --epochs 40 \
    --lr 0.001
```

**Quick test run (limited dataset for testing):**
```bash
python train_lenet.py \
    --data-dir ./data/mnist \
    --test-run \
    --test-samples 100 \
    --epochs 2 \
    --batch-size 64
```

Key features:
- Uses Adam optimizer
- Evaluates with precision, recall, and F1-macro metrics
- Input size: 32x32
- `--test-run`: Limits dataset size for quick testing (uses `--test-samples` for training, half for testing)

#### Bài 2: Train GoogLeNet on VinaFood21

```bash
python train_googlenet.py \
    --train-dir /path/to/vinafood21/train \
    --test-dir /path/to/vinafood21/test \
    --batch-size 32 \
    --epochs 30 \
    --lr 0.001
```

**CPU-only training (force CPU even if GPU available):**
```bash
CUDA_VISIBLE_DEVICES="" python train_googlenet.py \
    --train-dir /path/to/vinafood21/train \
    --test-dir /path/to/vinafood21/test \
    --batch-size 32 \
    --epochs 30 \
    --lr 0.001
```

**Quick test run (limited dataset for testing):**
```bash
python train_googlenet.py \
    --train-dir /path/to/vinafood21/train \
    --test-dir /path/to/vinafood21/test \
    --test-run \
    --test-samples 100 \
    --epochs 2 \
    --batch-size 16
```

Key features:
- First convolution layer has padding=3
- All MaxPooling layers use ceil_mode=True
- Uses Adam optimizer
- Evaluates with precision, recall, and F1 metrics
- Input size: 224x224
- `--test-run`: Limits dataset size for quick testing (uses `--test-samples` for training, half for testing)

#### Bài 3: Train ResNet-18 on VinaFood21

```bash
python train_resnet18.py \
    --train-dir /path/to/vinafood21/train \
    --test-dir /path/to/vinafood21/test \
    --batch-size 64 \
    --epochs 30 \
    --lr 0.001
```

**CPU-only training (force CPU even if GPU available):**
```bash
CUDA_VISIBLE_DEVICES="" python train_resnet18.py \
    --train-dir /path/to/vinafood21/train \
    --test-dir /path/to/vinafood21/test \
    --batch-size 64 \
    --epochs 30 \
    --lr 0.001
```

**Quick test run (limited dataset for testing):**
```bash
python train_resnet18.py \
    --train-dir /path/to/vinafood21/train \
    --test-dir /path/to/vinafood21/test \
    --test-run \
    --test-samples 100 \
    --epochs 2 \
    --batch-size 16
```

Key features:
- MaxPooling layers between residual blocks (kernel=3, stride=2, padding=0)
- Uses Adam optimizer
- Evaluates with precision, recall, and F1 metrics
- Input size: 224x224
- `--test-run`: Limits dataset size for quick testing (uses `--test-samples` for training, half for testing)

#### Bài 4: Fine-tune Pretrained ResNet50 on VinaFood21

```bash
python train_pretrained_resnet50.py \
    --train-dir /path/to/vinafood21/train \
    --test-dir /path/to/vinafood21/test \
    --batch-size 32 \
    --epochs 20 \
    --lr 1e-4
```

**CPU-only training (force CPU even if GPU available):**
```bash
CUDA_VISIBLE_DEVICES="" python train_pretrained_resnet50.py \
    --train-dir /path/to/vinafood21/train \
    --test-dir /path/to/vinafood21/test \
    --batch-size 32 \
    --epochs 20 \
    --lr 1e-4
```

**Quick test run (limited dataset for testing):**
```bash
python train_pretrained_resnet50.py \
    --train-dir /path/to/vinafood21/train \
    --test-dir /path/to/vinafood21/test \
    --test-run \
    --test-samples 100 \
    --epochs 2 \
    --batch-size 16
```

Key features:
- Uses pretrained ResNet50 from HuggingFace (microsoft/resnet-50)
- Fine-tuning with Adam optimizer
- Lower learning rate (1e-4) for transfer learning
- Evaluates with precision, recall, and F1 metrics
- Input size: 224x224
- `--test-run`: Limits dataset size for quick testing (uses `--test-samples` for training, half for testing)

### Multi-GPU Training (Distributed Data Parallel)

For maximum performance on multi-GPU systems, use `torchrun` to enable distributed training with DistributedDataParallel (DDP):

**Note**: Set `--nproc_per_node` to the number of GPUs you want to use (must not exceed available GPU count):
- For 2 GPUs: `--nproc_per_node=2`
- For 4 GPUs: `--nproc_per_node=4`  
- For 8 GPUs: `--nproc_per_node=8`

#### Bài 1: LeNet on Multiple GPUs

```bash
# For 2 GPUs:
torchrun --nproc_per_node=2 train_lenet.py \
    --data-dir ./data/mnist \
    --batch-size 256 \
    --epochs 40 \
    --lr 0.001

# For 4 GPUs:
torchrun --nproc_per_node=4 train_lenet.py \
    --data-dir ./data/mnist \
    --batch-size 256 \
    --epochs 40 \
    --lr 0.001
```

#### Bài 2: GoogLeNet on Multiple GPUs

```bash
# For 2 GPUs:
torchrun --nproc_per_node=2 train_googlenet.py \
    --train-dir /path/to/vinafood21/train \
    --test-dir /path/to/vinafood21/test \
    --batch-size 32 \
    --epochs 30 \
    --lr 0.001

# For any number of GPUs, adjust --nproc_per_node accordingly
```

#### Bài 3: ResNet-18 on Multiple GPUs

```bash
# For 2 GPUs:
torchrun --nproc_per_node=2 train_resnet18.py \
    --train-dir /path/to/vinafood21/train \
    --test-dir /path/to/vinafood21/test \
    --batch-size 64 \
    --epochs 30 \
    --lr 0.001

# For any number of GPUs, adjust --nproc_per_node accordingly
```

#### Bài 4: Pretrained ResNet50 on Multiple GPUs

```bash
# For 2 GPUs:
torchrun --nproc_per_node=2 train_pretrained_resnet50.py \
    --train-dir /path/to/vinafood21/train \
    --test-dir /path/to/vinafood21/test \
    --batch-size 32 \
    --epochs 20 \
    --lr 1e-4

# For any number of GPUs, adjust --nproc_per_node accordingly
```

**Notes on Distributed Training:**
- `--nproc_per_node` specifies the number of GPUs to use per node
- The scripts automatically work with any available GPU count - if no GPU is detected, training falls back to CPU
- Batch size is per GPU (effective batch size = batch_size × num_gpus)
- Learning rate is automatically scaled by the number of GPUs in distributed mode
- Model checkpoints are saved only on the main process (GPU 0)
- Progress bars and logs are shown only on the main process

### Performance Optimizations

The code includes several optimizations for efficient training:

1. **Automatic Device Detection**: Automatically uses best available hardware (CPU/GPU/Multi-GPU)
2. **DataParallel**: Automatically enabled for multiple GPUs without torchrun
3. **Mixed Precision Training (AMP)**: Automatic mixed precision for faster GPU training and reduced memory usage
4. **Distributed Data Parallel**: Maximum efficiency with `torchrun` for multi-GPU training
5. **Flexible Backend Selection**: NCCL for GPUs, Gloo for CPU-based distributed training
6. **Persistent Workers**: Data loader workers persist between epochs (in distributed mode)
7. **Pin Memory**: Faster data transfer to GPU
8. **Non-blocking Transfers**: Asynchronous GPU memory transfers
9. **Gradient Scaling**: Prevents underflow in mixed precision training

## Model Architecture Details

### LeNet
- 2 convolutional layers (5x5 kernels)
- Average pooling
- 3 fully connected layers
- ReLU activation

### GoogLeNet (Inception v1)
- Inception modules with parallel 1x1, 3x3, 5x5 convolutions
- Auxiliary classifiers for training
- Local Response Normalization
- All MaxPooling layers use ceil_mode=True

### ResNet-18
- 4 residual layer groups with 2 BasicBlocks each
- MaxPooling between groups (kernel=3, stride=2, padding=0)
- Skip connections
- Batch normalization

### Pretrained ResNet50
- HuggingFace pretrained model (microsoft/resnet-50)
- Modified classification head for VinaFood21 classes
- Transfer learning approach

## Output

All training scripts will:
1. Display training progress with loss and accuracy
2. Evaluate on test set after each epoch
3. Report precision, recall, and F1 scores
4. Save the best model to `./checkpoints/`

## Evaluation Metrics

All models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Macro-averaged precision
- **Recall**: Macro-averaged recall
- **F1 Score**: Macro-averaged F1 score

## Project Structure

```
ds201/
├── models/
│   ├── __init__.py
│   ├── lenet.py
│   ├── googlenet.py
│   ├── resnet.py
│   └── pretrained_resnet50.py
├── data_utils/
│   ├── __init__.py
│   ├── loaders.py
│   ├── training.py
│   └── evaluation.py
├── train_lenet.py
├── train_googlenet.py
├── train_resnet18.py
├── train_pretrained_resnet50.py
├── requirements.txt
└── README.md
```

## References

- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- VinaFood21 Dataset: https://arxiv.org/abs/2108.02929
- HuggingFace ResNet50: https://huggingface.co/microsoft/resnet-50
