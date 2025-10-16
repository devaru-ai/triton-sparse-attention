import matplotlib.pyplot as plt
import numpy as np

def plot_runtime_and_memory(results_dict, save_path='runtime_memory.png'):
    """
    Plot runtime and memory for different models (bar + line plot).
    """
    names = list(results_dict.keys())
    runtimes = [results_dict[n]['runtime_s'] for n in names]
    mem = [results_dict[n]['memory_mb'] for n in names]
    
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Runtime (s)', color=color)
    ax1.bar(names, runtimes, color=color, alpha=0.7)
    
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Memory (MB)', color=color)
    ax2.plot(names, mem, color=color, marker='o', linestyle='--')
    
    plt.title('Model Runtime & Memory Usage')
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_accuracy(results_dict, save_path='accuracy.png'):
    """
    Plot accuracy for different models as a bar chart.
    """
    names = list(results_dict.keys())
    accs = [results_dict[n]['accuracy'] for n in names]
    plt.figure()
    plt.bar(names, accs, color='tab:green')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.title('Model Accuracy Comparison')
    plt.savefig(save_path)
    plt.close()

def plot_attention_map(attn_matrix, save_path='attn_map.png'):
    """
    Plot an attention map (2D numpy or torch matrix).
    """
    if hasattr(attn_matrix, 'cpu'):  # Convert torch tensor to numpy if needed
        attn_matrix = attn_matrix.cpu().detach().numpy()
    plt.figure(figsize=(8,6))
    plt.imshow(attn_matrix, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Attention Heatmap')
    plt.xlabel('Key positions')
    plt.ylabel('Query positions')
    plt.savefig(save_path)
    plt.close()

def plot_scaling_trends(seq_lens, runtimes_dict, memories_dict, save_prefix='scaling'):
    """
    Plot scaling (runtime, memory) vs. sequence length for each model.
    Args:
        seq_lens: list of sequence lengths tested.
        runtimes_dict: {model_name: [runtime_seq1, runtime_seq2, ... ]}
        memories_dict: {model_name: [memory_seq1, memory_seq2, ... ]}
        save_prefix: beginning of plot file names.
    """
    plt.figure()
    for name, runtimes in runtimes_dict.items():
        plt.plot(seq_lens, runtimes, marker='o', label=name)
    plt.xlabel('Sequence Length')
    plt.ylabel('Runtime (s)')
    plt.title('Runtime Scaling vs. Sequence Length')
    plt.legend()
    plt.savefig(f'{save_prefix}_runtime_scaling.png')
    plt.close()

    plt.figure()
    for name, memories in memories_dict.items():
        plt.plot(seq_lens, memories, marker='o', label=name)
    plt.xlabel('Sequence Length')
    plt.ylabel('Peak Memory (MB)')
    plt.title('Memory Usage Scaling vs. Sequence Length')
    plt.legend()
    plt.savefig(f'{save_prefix}_memory_scaling.png')
    plt.close()
