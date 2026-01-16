import subprocess
import re
import matplotlib.pyplot as plt
import os
import sys

def run_executable(exe_name):
    """Runs the executable and returns STDOUT output."""
    if not os.path.exists(exe_name):
        print(f"Error: {exe_name} not found. Please compile it first.")
        return None
    
    print(f"Running {exe_name}...")
    try:
        result = subprocess.run([exe_name], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running {exe_name}: {e}")
        return None

def parse_output(output):
    """Parses the output text for timing metrics."""
    metrics = {}
    
    patterns = {
        'Stack Push': r"Stack Push Time:\s+([\d\.]+)\s+ms",
        'Stack Pop': r"Stack Pop Time:\s+([\d\.]+)\s+ms",
        'Queue Enqueue': r"Queue Enqueue Time:\s+([\d\.]+)\s+ms",
        'Queue Dequeue': r"Queue Dequeue Time:\s+([\d\.]+)\s+ms",
        'Sequential Stack': r"Sequential Stack Time:\s+([\d\.]+)\s+ms",
        'Sequential Queue': r"Sequential Queue Time:\s+([\d\.]+)\s+ms",
        'MPMC Queue': r"MPMC Queue Time:\s+([\d\.]+)\s+ms",
        'Shared Mem Stack': r"Shared Memory Stack Time:\s+([\d\.]+)\s+ms"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))
            
    return metrics

def plot_results(metrics):
    """Generates bar charts for the metrics."""
    if not metrics:
        print("No metrics to plot.")
        return

    # Categories for plotting
    stack_keys = ['Stack Push', 'Stack Pop', 'Sequential Stack']
    queue_keys = ['Queue Enqueue', 'Queue Dequeue', 'Sequential Queue']
    opt_keys = ['MPMC Queue', 'Shared Mem Stack'] 
    
    # Filter available metrics
    stack_data = {k: metrics.get(k, 0) for k in stack_keys if k in metrics}
    queue_data = {k: metrics.get(k, 0) for k in queue_keys if k in metrics}
    # For optimization comparison, let's compare standard vs optimized
    # Total Parallel Queue (Enq+Deq) vs MPMC
    # Total Parallel Stack (Push+Pop) vs Shared Mem Stack (simulated)
    
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Stack Performance
    if stack_data:
        axes[0].bar(stack_data.keys(), stack_data.values(), color=['skyblue', 'steelblue', 'orange'])
        axes[0].set_title('Stack Performance (Lower is Better)')
        axes[0].set_ylabel('Time (ms)')
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        for i, v in enumerate(stack_data.values()):
            axes[0].text(i, v, f"{v:.2f}", ha='center', va='bottom')

    # Plot 2: Queue Performance
    if queue_data:
        axes[1].bar(queue_data.keys(), queue_data.values(), color=['lightgreen', 'forestgreen', 'orange'])
        axes[1].set_title('Queue Performance (Lower is Better)')
        axes[1].set_ylabel('Time (ms)')
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        for i, v in enumerate(queue_data.values()):
            axes[1].text(i, v, f"{v:.2f}", ha='center', va='bottom')

    # Plot 3: Optimizations vs Standard (Approximation)
    # We compare Enqueue vs MPMC (since MPMC kernel does both, but mostly dominated by atomics)
    # Actually MPMC output says "Enqueue + Dequeue in one kernel"
    # So we should sum Parallel Queue Enqueue + Dequeue to compare fairly
    
    opt_comparison = {}
    if 'Queue Enqueue' in metrics and 'Queue Dequeue' in metrics:
        opt_comparison['Std Queue (Enq+Deq)'] = metrics['Queue Enqueue'] + metrics['Queue Dequeue']
    if 'MPMC Queue' in metrics:
        opt_comparison['MPMC Queue (Enq+Deq)'] = metrics['MPMC Queue']
        
    if 'Stack Push' in metrics and 'Stack Pop' in metrics:
         opt_comparison['Std Stack (Push+Pop)'] = metrics['Stack Push'] + metrics['Stack Pop']
    if 'Shared Mem Stack' in metrics:
        opt_comparison['Shared Stack'] = metrics['Shared Mem Stack']

    if opt_comparison:
        colors = []
        for k in opt_comparison.keys():
            if 'Std' in k: colors.append('gray')
            elif 'MPMC' in k: colors.append('lightgreen')
            else: colors.append('skyblue') # Shared Stack
            
        axes[2].bar(opt_comparison.keys(), opt_comparison.values(), color=colors)
        axes[2].set_title('Standard vs Optimized (Lower is Better)')
        axes[2].set_ylabel('Time (ms)')
        axes[2].grid(axis='y', linestyle='--', alpha=0.7)
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=15)
        for i, v in enumerate(opt_comparison.values()):
            axes[2].text(i, v, f"{v:.2f}", ha='center', va='bottom')

    plt.tight_layout()
    output_file = 'comparison_results.png'
    plt.savefig(output_file)
    print(f"Plot saved to {os.path.abspath(output_file)}")

def main():
    # Detect executables (assuming windows .exe extension based on previous context)
    base_exe = "stack_queue.exe"
    opt_exe = "stack_queue_opt.exe"
    
    output_text = ""
    
    # Run Basic
    out1 = run_executable(base_exe)
    if out1: output_text += out1
    
    # Run Optimized
    out2 = run_executable(opt_exe)
    if out2: output_text += out2
    
    if not output_text:
        print("No output generated. Exiting.")
        return

    # Parse
    metrics = parse_output(output_text)
    
    # Print parsed metrics for verification
    print("Parsed Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v} ms")
        
    # Plot
    plot_results(metrics)

if __name__ == "__main__":
    main()
