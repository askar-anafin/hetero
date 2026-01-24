import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results():
    print("Reading results.csv...")
    if not os.path.exists("results.csv"):
        print("Error: results.csv not found. Run the C++ project first.")
        return

    df = pd.read_csv("results.csv")
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(df['Size'], df['CPU_Seq'], marker='o', label='CPU Sequential (std::sort)')
    plt.plot(df['Size'], df['CPU_OMP'], marker='s', label='CPU Parallel (OpenMP)')
    plt.plot(df['Size'], df['GPU_Bitonic'], marker='^', label='GPU Bitonic (CUDA)')
    
    plt.xlabel('Array Size')
    plt.ylabel('Time (ms)')
    plt.title('Benchmark Results: CPU vs GPU Sorting')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    # Use log scale for x and y axes if the range is large (which it is, powers of 2)
    plt.xscale('log', base=2)
    plt.yscale('log')
    
    output_file = 'benchmark_plot.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    
    # Try to open the image automatically on Windows
    os.system(f"start {output_file}")

if __name__ == "__main__":
    plot_results()
