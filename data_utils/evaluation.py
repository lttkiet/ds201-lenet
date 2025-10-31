
import torch
import torch.distributed as dist
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def evaluate_model(model, test_loader, criterion, device, is_distributed=False):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    use_cuda = device.type == 'cuda'

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_cuda:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.detach().item() * labels.size(0)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if is_distributed:
        loss_tensor = torch.tensor([val_loss, len(all_labels)], 
                                   dtype=torch.float32, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        val_loss = loss_tensor[0].item() / loss_tensor[1].item()

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        preds_tensor = torch.from_numpy(all_preds).to(device)
        labels_tensor = torch.from_numpy(all_labels).to(device)

        local_size = torch.tensor([len(all_preds)], dtype=torch.long, device=device)
        size_list = [torch.zeros(1, dtype=torch.long, device=device) 
                     for _ in range(world_size)]
        dist.all_gather(size_list, local_size)

        max_size = max([s.item() for s in size_list])

        if len(all_preds) < max_size:
            pad_size = max_size - len(all_preds)
            preds_tensor = torch.cat([preds_tensor, 
                                     torch.zeros(pad_size, dtype=preds_tensor.dtype, 
                                               device=device)])
            labels_tensor = torch.cat([labels_tensor, 
                                      torch.zeros(pad_size, dtype=labels_tensor.dtype, 
                                                device=device)])

        gathered_preds = [torch.zeros(max_size, dtype=torch.long, device=device) 
                         for _ in range(world_size)]
        gathered_labels = [torch.zeros(max_size, dtype=torch.long, device=device) 
                          for _ in range(world_size)]

        dist.all_gather(gathered_preds, preds_tensor)
        dist.all_gather(gathered_labels, labels_tensor)

        all_preds_list = []
        all_labels_list = []
        for i, size in enumerate(size_list):
            size = size.item()
            all_preds_list.append(gathered_preds[i][:size].cpu().numpy())
            all_labels_list.append(gathered_labels[i][:size].cpu().numpy())

        all_preds = np.concatenate(all_preds_list)
        all_labels = np.concatenate(all_labels_list)
    else:
        val_loss = val_loss / len(all_labels) if len(all_labels) > 0 else 0

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    return val_loss, accuracy, precision, recall, f1

def print_evaluation(val_loss, accuracy, precision, recall, f1):
    print('\n' + '=' * 60)
    print('EVALUATION RESULTS')
    print('=' * 60)
    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('=' * 60)