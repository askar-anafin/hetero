# Import matplotlib for creating plots and visualizations
import matplotlib.pyplot as plt
# Import pandas for data manipulation and CSV file reading
import pandas as pd
# Import numpy for numerical operations
import numpy as np

# Read the CSV file containing benchmark results into a DataFrame
df = pd.read_csv('results.csv')

# Create a figure with 2x2 grid of subplots (4 plots total), size 14x10 inches
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# Add a main title for the entire figure with bold font
fig.suptitle('CUDA Performance Analysis: Reduction and Sorting', fontsize=16, fontweight='bold')

# Plot 1: Reduction comparison (Global vs Shared)
# Access the first subplot (top-left) from the 2x2 grid
ax1 = axes[0, 0]
# Create x-axis positions for the bars (0, 1, 2, ... for each data row)
x = np.arange(len(df['Size']))
# Set the width of each bar
width = 0.35

# Create first set of bars for Global Memory reduction, shifted left by half width
# Color is red (#e74c3c), transparency 0.8
bars1 = ax1.bar(x - width/2, df['ReductionGlobal'], width, label='Global Memory', color='#e74c3c', alpha=0.8)
# Create second set of bars for Shared Memory reduction, shifted right by half width
# Color is blue (#3498db), transparency 0.8
bars2 = ax1.bar(x + width/2, df['ReductionShared'], width, label='Shared Memory', color='#3498db', alpha=0.8)

# Set the x-axis label with bold font
ax1.set_xlabel('Array Size', fontweight='bold')
# Set the y-axis label with bold font
ax1.set_ylabel('Time (ms)', fontweight='bold')
# Set the title for this subplot
ax1.set_title('Reduction: Global vs Shared Memory')
# Set the x-axis tick positions to match our data points
ax1.set_xticks(x)
# Set the x-axis labels to formatted array sizes with thousand separators
ax1.set_xticklabels([f'{int(size):,}' for size in df['Size']])
# Display the legend showing which color represents which memory type
ax1.legend()
# Add a grid for easier reading, with 30% transparency
ax1.grid(True, alpha=0.3)

# Add value labels on top of each bar
# Iterate through both bar groups (bars1 and bars2)
for bars in [bars1, bars2]:
    # For each individual bar in the group
    for bar in bars:
        # Get the height (value) of the bar
        height = bar.get_height()
        # Add text label centered on the bar, just above its top
        # Position: (center of bar horizontally, top of bar vertically)
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',  # Format value to 2 decimal places
                ha='center', va='bottom', fontsize=8)  # Horizontal center, vertical bottom alignment

# Plot 2: Speedup (Shared vs Global)
# Access the second subplot (top-right) from the 2x2 grid
ax2 = axes[0, 1]
# Calculate speedup factor: time with global / time with shared memory
speedup = df['ReductionGlobal'] / df['ReductionShared']
# Create bars showing speedup values with green color (#2ecc71)
bars3 = ax2.bar(x, speedup, color='#2ecc71', alpha=0.8)

# Set the x-axis label with bold font
ax2.set_xlabel('Array Size', fontweight='bold')
# Set the y-axis label with bold font
ax2.set_ylabel('Speedup Factor', fontweight='bold')
# Set the title for this subplot
ax2.set_title('Reduction Speedup (Shared vs Global)')
# Set the x-axis tick positions to match our data points
ax2.set_xticks(x)
# Set the x-axis labels to formatted array sizes with thousand separators
ax2.set_xticklabels([f'{int(size):,}' for size in df['Size']])
# Add a horizontal red dashed line at y=1 to show the "no speedup" baseline
ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No speedup')
# Display the legend
ax2.legend()
# Add a grid for easier reading, with 30% transparency
ax2.grid(True, alpha=0.3)

# Add value labels on top of bars showing speedup factor
# For each bar in the speedup chart
for bar in bars3:
    # Get the height (speedup value) of the bar
    height = bar.get_height()
    # Add text label centered on the bar, just above its top
    # Format with "x" suffix to indicate speedup multiplier
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}x',  # Format to 2 decimal places with "x" suffix
            ha='center', va='bottom', fontsize=9, fontweight='bold')  # Bold text

# Plot 3: Sorting time vs array size
# Access the third subplot (bottom-left) from the 2x2 grid
ax3 = axes[1, 0]
# Create a line plot with circular markers for sorting performance
# Line width 2, marker size 8, purple color (#9b59b6)
ax3.plot(df['Size'], df['Sorting'], marker='o', linewidth=2, markersize=8, 
         color='#9b59b6', label='Sorting Time')
# Fill the area under the line with semi-transparent purple (alpha=0.3)
ax3.fill_between(df['Size'], df['Sorting'], alpha=0.3, color='#9b59b6')

# Set the x-axis label with bold font
ax3.set_xlabel('Array Size', fontweight='bold')
# Set the y-axis label with bold font
ax3.set_ylabel('Time (ms)', fontweight='bold')
# Set the title for this subplot
ax3.set_title('Sorting Performance vs Array Size')
# Use logarithmic scale for x-axis (better for wide range of sizes)
ax3.set_xscale('log')
# Use logarithmic scale for y-axis (better for wide range of times)
ax3.set_yscale('log')
# Display the legend
ax3.legend()
# Add grid for both major and minor ticks, with 30% transparency
ax3.grid(True, alpha=0.3, which='both')

# Format x-axis labels to show actual array sizes with thousand separators
# Set tick positions to the actual size values from data
ax3.set_xticks(df['Size'])
# Set tick labels with formatted numbers
ax3.set_xticklabels([f'{int(size):,}' for size in df['Size']])

# Plot 4: All operations comparison
# Access the fourth subplot (bottom-right) from the 2x2 grid
ax4 = axes[1, 1]
# Plot Reduction (Global Memory) with square markers, line width 2
# Red color (#e74c3c), marker size 7
ax4.plot(df['Size'], df['ReductionGlobal'], marker='s', linewidth=2, 
         markersize=7, label='Reduction (Global)', color='#e74c3c')
# Plot Reduction (Shared Memory) with circular markers, line width 2
# Blue color (#3498db), marker size 7
ax4.plot(df['Size'], df['ReductionShared'], marker='o', linewidth=2, 
         markersize=7, label='Reduction (Shared)', color='#3498db')
# Plot Sorting with triangle markers, line width 2
# Purple color (#9b59b6), marker size 7
ax4.plot(df['Size'], df['Sorting'], marker='^', linewidth=2, 
         markersize=7, label='Sorting', color='#9b59b6')

# Set the x-axis label with bold font
ax4.set_xlabel('Array Size', fontweight='bold')
# Set the y-axis label with bold font
ax4.set_ylabel('Time (ms)', fontweight='bold')
# Set the title for this subplot
ax4.set_title('Overall Performance Comparison')
# Use logarithmic scale for x-axis
ax4.set_xscale('log')
# Use logarithmic scale for y-axis
ax4.set_yscale('log')
# Display the legend showing all three operations
ax4.legend()
# Add grid for both major and minor ticks, with 30% transparency
ax4.grid(True, alpha=0.3, which='both')

# Format x-axis labels to show actual array sizes with thousand separators
# Set tick positions to the actual size values from data
ax4.set_xticks(df['Size'])
# Set tick labels with formatted numbers
ax4.set_xticklabels([f'{int(size):,}' for size in df['Size']])

# Adjust the layout to prevent overlapping of subplots and labels
plt.tight_layout()

# Save the figure to a PNG file with 300 DPI (high quality), tight bounding box
plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
# Print confirmation message to console
print("Performance graph saved as 'performance_analysis.png'")

# Create a summary table in console output
# Print section header
print("\n=== Performance Summary ===")
# Print the entire DataFrame as a formatted table without row indices
print(df.to_string(index=False))
# Print speedup analysis section header
print("\n=== Speedup Analysis ===")
# Print table header with column names and spacing
print(f"{'Array Size':<15} {'Speedup (Shared vs Global)':<30}")
# Print separator line
print("-" * 45)
# Iterate through each row in the DataFrame with index
for i, row in df.iterrows():
    # Calculate speedup: Global time / Shared time
    speedup = row['ReductionGlobal'] / row['ReductionShared']
    # Print formatted row: Size with thousand separator, speedup with 2 decimals and "x"
    print(f"{int(row['Size']):<15,} {speedup:.2f}x")

# Display the plot window (interactive mode)
plt.show()
