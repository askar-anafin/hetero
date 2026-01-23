# Heterogeneous Parallelization in C++

This repository contains a collection of assignments and practical works focused on Heterogeneous Computing, Parallel Programming, and High-Performance Computing using C++, OpenMP, CUDA, MPI, and OpenCL.

## 📂 Repository Structure

The repository is divided into two main sections:

- **Assignments**: Major distinct tasks covering key parallelization concepts.
- **Practices**: Hands-on laboratory works and smaller practice tasks.

### 📝 Assignments

| Assignment | Description | Technologies |
|------------|-------------|--------------|
| **Assignment 1** | **Introduction to Parallelization**<br>Basics of C++ memory management, sequential vs. parallel search (Min/Max), and Parallel Average calculation using OpenMP Reductions. | C++, OpenMP |
| **Assignment 2** | **Heterogeneous Computing Fundamentals**<br>Theoretical/Practical tasks on CPU vs GPU architecture. Includes Parallel Min/Max (OpenMP), Selection Sort (OpenMP), and Merge Sort (CUDA). | C++, OpenMP, CUDA |
| **Assignment 3** | **CUDA Programming Basics**<br>Implementation of basic CUDA kernels, memory management, and thread hierarchy. | C++, CUDA |
| **Assignment 4** | **Hybrid Computing (MPI + CUDA)**<br>Advanced distributed computing task involving MPI for process communication and CUDA for local node acceleration. Includes distributed sorting algorithms. | C++, MPI (MS-MPI), CUDA |

### 🔬 Practices

| Practice | Topic | Description |
|----------|-------|-------------|
| **Practice 1-3** | **Basics** | Introductory labs for C++ and CUDA environment setup. |
| **Practice 4** | **Parallel Algorithms** | Implementation of parallel reduction, scanning (prefix sum), and sorting algorithms on GPU. |
| **Practice 5** | **Data Structures** | Implementation of Parallel Stack and Queue data structures on GPU using Atomic Operations. |
| **Practice 6** | **OpenCL** | Introduction to OpenCL framework. Includes **Vector Addition** and **Matrix Multiplication** tasks. |
| **Practice 7** | **Scan & Reduction** | Advanced optimization of prefix sums and reduction algorithms on GPU. |
| **Practice 8-10**| **Advanced Topics** | PDF descriptions for advanced practical works. |

---

## 🛠 Prerequisites

To build and run the projects in this repository, you will need the following tools:

1. **C++ Compiler**: `g++` (MinGW) or `cl.exe` (MSVC) with C++11 support or higher.
2. **CUDA Toolkit**: Required for compiling `.cu` files and running CUDA applications (NVCC compiler).
3. **OpenMP**: Usually included with GCC/MSVC. Ensure your compiler supports it (e.g., `-fopenmp` for GCC).
4. **MS-MPI**: Microsoft MPI SDK and Runtime are required for **Assignment 4**.
5. **OpenCL SDK**: Required for **Practice 6** (typically included with NVIDIA CUDA Toolkit or GPU drivers).

## 🚀 How to Build and Run

### Standard C++/OpenMP

```bash
g++ main.cpp -o app -fopenmp
./app
```

### CUDA

```bash
nvcc main.cu -o app
./app
```

### Assignment 4 (Hybrid MPI + CUDA)

Navigate to the `assignments/assignment4` directory and use the provided build script:

```cmd
cd assignments/assignment4
build.bat
```

*Note: This usually requires `mpiexec` to run the distributed executable.*

### Practice 6 (OpenCL)

Practice 6 typically requires linking against OpenCL. Check the `start.bat` or `CMakeLists.txt` inside the folder for specific build instructions.

## 📚 Topics Covered

- **Parallel Patterns**: Map, Reduce, Scan, Scatter, Gather.
- **Memory Models**: Shared Memory, Global Memory, Constant Memory.
- **Synchronization**: Atomics, Barriers, Mutexes (CPU).
- **Heterogeneous Architectures**: CPU (Latency-oriented) vs GPU (Throughput-oriented).
- **APIs**: NVIDIA CUDA, OpenMP, MPI, OpenCL.
