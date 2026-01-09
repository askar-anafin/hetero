# Practice 5: Parallel Stack and Queue on CUDA

This project implements thread-safe stack and queue data structures for CUDA kernels using atomic operations.

## Features

1. **Parallel Stack**:
    - Uses `atomicAdd` and `atomicSub` for `push` and `pop`.
    - Handles multiple threads concurrently.
2. **Parallel Queue**:
    - Uses `atomicAdd` for `enqueue` and `dequeue`.
    - MPMC (Multi-Producer Multi-Consumer) support.
3. **Optimizations**:
    - Shared memory implementation for block-local operations.
    - Benchmarking comparison with sequential C++ versions.

## Files

- `stack_queue.cu`: Basic implementation and comparison vs sequential.
- `stack_queue_opt.cu`: MPMC queue and shared memory optimization.

## Performance Summary (N=1,000,000)

- **CUDA Queue** is highly efficient for large parallel workloads.
- **Shared Memory** reduces latency significantly for local operations.
- **Sequential** versions are bottlenecked by single-thread CPU performance when `N` is large.

## How to Run

Ensure `nvcc` and a C++ compiler (Visual Studio on Windows) are in your PATH.

```bash
nvcc stack_queue.cu -o stack_queue
./stack_queue
```
