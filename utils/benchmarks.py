import torch
import time
from torch.utils.data import DataLoader, TensorDataset

def time_model(model, batch, device='cuda', n_runs=10):
    """
    Measure average runtime for a forward pass.
    """
    model = model.to(device)
    batch = batch.to(device)
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.time()
            _ = model(batch)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
    avg_time = sum(times) / len(times)
    return avg_time

def measure_memory(model, batch, device='cuda'):
    """
    Measure peak memory usage for a forward pass.
    """
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    model = model.to(device)
    batch = batch.to(device)
    model.eval()
    with torch.no_grad():
        _ = model(batch)
    mem = torch.cuda.max_memory_allocated() / 1024**2 if device == 'cuda' else 0.0
    return mem

def compute_accuracy(model, data_loader, device='cuda'):
    """
    Compute accuracy (for token prediction/input-output mapping) across a dataloader.
    Assumes model produces logits for token predictions and integer target labels.
    """
    model = model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / total if total > 0 else 0.0

def benchmark_all(models, inputs, targets, batch_size=32, device='cuda'):
    """
    Run runtime, memory, and accuracy for a dict of models, using structured (inputs, targets) data.
    """
    # For consistency, use the same synthetic batch (first batch_size examples)
    batch = torch.from_numpy(inputs[:batch_size]).long().to(device)  # For timing/memory
    # Set up DataLoader for accuracy (process all data, batched)
    tensor_dataset = TensorDataset(torch.from_numpy(inputs).long(), torch.from_numpy(targets).long())
    data_loader = DataLoader(tensor_dataset, batch_size=batch_size)
    results = {}
    for name, model in models.items():
        time_sec = time_model(model, batch, device)
        mem_mb = measure_memory(model, batch, device)
        acc = compute_accuracy(model, data_loader, device)
        results[name] = {'runtime_s': time_sec, 'memory_mb': mem_mb, 'accuracy': acc}
        print(f"{name}: time={time_sec:.4f}s, mem={mem_mb:.1f}MB, acc={acc:.3f}")
    return results
