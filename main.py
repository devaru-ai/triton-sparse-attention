import os
import numpy as np
import torch

from utils.benchmarks import time_model, measure_memory

from models.vanilla_transformer import build_vanilla_transformer
from models.sparse_window import build_sparse_window_transformer
from models.sparse_window_vectorized import build_sparse_window_vectorized_transformer
from models.triton_sparse_window import build_triton_sparse_window_transformer

def ensure_cuda():
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
        return 'cuda'
    else:
        print("CUDA not available, using CPU")
        return 'cpu'

def generate_structured_data(seq_len, num_sequences=1, vocab_size=100):
    # For copy task or memory-only test
    inputs = np.random.randint(1, vocab_size+1, size=(num_sequences, seq_len), dtype=np.int32)
    targets = inputs.copy()
    return inputs, targets

def main():
    device = ensure_cuda()
    results_dir = 'experiments/results/'
    os.makedirs(results_dir, exist_ok=True)

    # Use very large sequence lengths to force quadratic scaling
    seq_lens = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    batch_size = 1
    d_model = 64
    num_layers = 1

    model_names = ['vanilla', 'window', 'window_vectorized', 'window_triton']
    runtimes = {name: [] for name in model_names}
    memories = {name: [] for name in model_names}
    failed = {name: [] for name in model_names}

    for seq_len in seq_lens:
        print(f"\n===== BENCHMARKING SEQUENCE LENGTH {seq_len} =====")
        inputs, targets = generate_structured_data(seq_len, num_sequences=batch_size)
        vocab_size = int(inputs.max() + 1)

        models = {
            'vanilla': build_vanilla_transformer(vocab_size, d_model=d_model, num_layers=num_layers, max_seq_len=seq_len),
            'window': build_sparse_window_transformer(vocab_size, d_model=d_model, num_layers=num_layers, window_size=2, max_seq_len=seq_len),
            'window_vectorized': build_sparse_window_vectorized_transformer(vocab_size, d_model=d_model, num_layers=num_layers, window_size=2, max_seq_len=seq_len),
            'window_triton': build_triton_sparse_window_transformer(vocab_size, d_model=d_model, num_layers=num_layers, window_size=2, max_seq_len=seq_len),
        }

        for name, model in models.items():
            try:
                print(f"Timing {name}...", end='')
                runtime = time_model(model, torch.from_numpy(inputs).long(), device=device, n_runs=3)
                mem = measure_memory(model, torch.from_numpy(inputs).long(), device=device)
                print(f" time={runtime:.4f}s, mem={mem:.1f}MB")
                runtimes[name].append(runtime)
                memories[name].append(mem)
                failed[name].append(False)
            except RuntimeError as e:
                print(f" {name} OOM or error ({str(e)[:50]})")
                runtimes[name].append(None)
                memories[name].append(None)
                failed[name].append(True)
                torch.cuda.empty_cache()

    # --- Plot scaling trends ---
    import matplotlib.pyplot as plt
    seq_lens = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    #pretty_labels = ["512", "1k", "2k", "4k", "8k", "16k", "32k"]

    plt.figure()
    for name in runtimes:
        xvals = [l for l, f in zip(seq_lens, failed[name]) if not f]
        yvals = [t for t, f in zip(runtimes[name], failed[name]) if not f]
        plt.plot(xvals, yvals, marker="o", label=name)
    plt.xlabel("Sequence Length")
    plt.ylabel("Runtime (s)")
    #plt.title("Sparse vs Dense Attention: Runtime Scaling")
    plt.legend()
    #plt.xticks(seq_lens, pretty_labels) 
    plt.savefig(os.path.join(results_dir, "sparse_vs_dense_runtime_scaling.png"))
    plt.close()

    plt.figure()
    for name in memories:
        xvals = [l for l, f in zip(seq_lens, failed[name]) if not f]
        yvals = [m for m, f in zip(memories[name], failed[name]) if not f]
        plt.plot(xvals, yvals, marker="o", label=name)
    plt.xlabel("Sequence Length")
    plt.ylabel("Peak Memory (MB)")
    #plt.title("Sparse vs Dense Attention: Memory Scaling")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "sparse_vs_dense_memory_scaling.png"))
    plt.close()

    print(f"\nScaling plots saved to {results_dir}")

if __name__ == "__main__":
    main()
