# Assignment 4 Report

## Task 1: CUDA Array Sum (Global Memory)

- **Objective**: Compute sum of 100,000 integers.
- **Results**:
  - CPU Time: ~0.12 ms
  - GPU Time: ~0.15 ms
  - Result: **PASSED**
- **Observation**: For small arrays (100k), CPU and GPU performance is comparable, with CPU often being faster due to kernel launch overhead. The GPU's massive parallelism shines with significantly larger datasets.

## Task 2: CUDA Prefix Sum (Shared Memory)

- **Objective**: Compute prefix sum (scan) of 1,000,000 integers.
- **Results**:
  - CPU Time: ~1.54 ms
  - GPU Time: ~0.63 ms
  - Result: **PASSED**
- **Observation**: The GPU (Shared Memory implementation) is ~2.4x faster than the CPU for 1M elements. This demonstrates the efficiency of using shared memory to reduce global memory latency in bandwidth-bound algorithms like Scan.

## Task 3: Hybrid CPU + GPU Processing

- **Objective**: Process 10,000,000 floats with heavy computation (`sin*cos+sqrt`) split between CPU (50%) and GPU (50%).
- **Results**:
  - Pure CPU: ~151.4 ms
  - Pure GPU: ~2.7 ms
  - Hybrid (50-50 split): ~74.8 ms
  - Result: **PASSED**
- **Observation**:
  - Pure GPU is drastically faster (~56x) than Pure CPU for this compute-bound task.
  - The Hybrid approach time (~75 ms) is roughly half of the CPU time because the CPU's 50% chunk limits the overall speed (Amdahl's Law). The GPU finishes its 50% chunk almost instantly (~1.35ms) and then waits for the CPU.
  - **Conclusion**: For heterogeneous computing to be effective here, the load balance needs adjustment (e.g., 98% GPU, 2% CPU) or the dataset needs to be larger than GPU memory.

## Task 4: Distributed MPI Program

- **Objective**: Distributed array processing (Multiplying 10M elements by 2) using `MPI_Scatter` and `MPI_Gather`.
- **Results**:
  - **2 Processes**: ~29.34 ms
  - **4 Processes**: ~22.81 ms
  - **8 Processes**: ~19.25 ms
  - Result: **PASSED**
- **Observation**:
  - We observe parallel speedup as process count increases:
    - 2 -> 4 processes: ~1.3x speedup.
    - 4 -> 8 processes: ~1.2x speedup.
  - The scalability is not perfectly linear (e.g., 2x speedup for 2x processes) due to the overhead of inter-process communication (Scatter/Gather of 40MB data) and process creation on Windows.
