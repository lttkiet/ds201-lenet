
import torch
import torch.distributed as dist
import time
from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None, 
               is_distributed=False, world_size=1, sampler=None, epoch=0):
    model.train()

    if is_distributed and sampler is not None:
        sampler.set_epoch(epoch)

    running_loss = 0.0
    correct = 0
    total = 0

    use_cuda = device.type == 'cuda'

    show_progress = not is_distributed or dist.get_rank() == 0
    iterator = tqdm(train_loader, desc='Training', disable=not show_progress)

    for images, labels in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and use_cuda:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)

                if isinstance(outputs, tuple):
                    main_out, aux1_out, aux2_out = outputs
                    loss = criterion(main_out, labels) + \
                           0.3 * criterion(aux1_out, labels) + \
                           0.3 * criterion(aux2_out, labels)
                    outputs = main_out
                else:
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)

            if isinstance(outputs, tuple):
                main_out, aux1_out, aux2_out = outputs
                loss = criterion(main_out, labels) + \
                       0.3 * criterion(aux1_out, labels) + \
                       0.3 * criterion(aux2_out, labels)
                outputs = main_out
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        running_loss += loss.detach().item() * labels.size(0)
        preds = outputs.detach().argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    if is_distributed:
        metrics = torch.tensor([running_loss, correct, total], 
                              dtype=torch.float32, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        running_loss, correct, total = metrics.tolist()

    epoch_loss = running_loss / total if total > 0 else 0
    epoch_accuracy = correct / total if total > 0 else 0

    return epoch_loss, epoch_accuracy

def train_model(model, train_loader, test_loader, criterion, optimizer, 
                device, epochs=40, scaler=None, save_path=None, is_distributed=False,
                world_size=1, sampler=None, early_stopping_patience=None):
    from .evaluation import evaluate_model

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    best_f1 = 0.0
    epochs_without_improvement = 0
    is_main = not is_distributed or dist.get_rank() == 0

    for epoch in range(epochs):
        if is_main:
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print('-' * 60)

        start_time = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            is_distributed=is_distributed, world_size=world_size, 
            sampler=sampler, epoch=epoch
        )
        train_time = time.time() - start_time

        val_loss, val_acc, precision, recall, f1 = evaluate_model(
            model, test_loader, criterion, device, is_distributed=is_distributed
        )

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1'].append(f1)

        if is_main:
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
            print(f'Time: {train_time:.2f}s')

        if save_path and f1 > best_f1 and is_main:
            best_f1 = f1
            epochs_without_improvement = 0
            if hasattr(model, 'module'):
                model_to_save = model.module
            else:
                model_to_save = model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': f1,
            }, save_path)
            print(f'Saved best model with F1: {f1:.4f}')
        else:
            epochs_without_improvement += 1

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            if is_main:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                print(f'No improvement for {early_stopping_patience} consecutive epochs')
                print(f'Best F1 score: {best_f1:.4f}')
            break

    return history