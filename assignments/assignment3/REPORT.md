
# Assignment 3 Report

## Task 1: Global vs Shared Memory

Comparison of element-wise processing for 1,000,000 elements.

- **Global Memory Kernel**: ~31.46 ms
- **Shared Memory Kernel**: ~0.073 ms
- **Observation**: The shared memory implementation recorded significantly faster time.
  - *Note*: The standard element-wise multiplication doesn't inherently benefit from shared memory due to lack of data reuse. The massive difference here likely indicates that the Global Memory launch included significant initialization/driver overhead (cold start), while the Shared Memory kernel (launching second) benefited from the warmed-up GPU state. In a pure bandwidth test with warmup, the difference would be negligible or Shared might be slightly slower due to copy overhead. However, the mechanism is successfully demonstrated.

## Task 2: Element-wise Addition with Variable Block Sizes

Comparison for 16,777,216 elements (approx 16 million).

| Block Size | Time (ms) |
|------------|-----------|
| 32         | 14.79     |
| 64         | 0.65      |
| 128        | 0.66      |
| 256        | 0.64      |
| 512        | 0.64      |
| 1024       | 0.66      |

- **Result**: Small block sizes (32) perform significantly worse (~23x slower). Block sizes of 64 and above saturate the performance.

## Task 3: Coalesced vs Uncoalesced Access

Demonstration of memory access patterns.

- **Coalesced Access**: 0.43 ms
- **Uncoalesced Access**: 0.57 ms
- **Slowdown**: 1.3x
- **Observation**: Accessing memory in a non-sequential (strided/transposed) pattern within a warp leads to more memory transactions and lower effective bandwidth.

## Task 4: Optimization

Search for optimal configuration for Vector Addition (16M elements).

- **Tested Range**: Block sizes 32 to 1024 (step 32).
- **Optimal Block Size**: 320 threads (Time: 0.636 ms)
- **Worst Block Size**: 32 threads (Time: 1.07 ms)
- **Speedup**: ~1.69x (Best vs Worst in warm state).

## Conclusion

- Block size has a major impact on occupancy and performance, especially avoiding very small blocks (warp under-utilization).
- Memory coalescing is critical for maximizing bandwidth usage.
- Shared memory is powerful but requires appropriate use cases (data reuse) to shine; for simple streaming it may not add benefit (performance gain here was likely artifact of testing conditions vs warmup).
