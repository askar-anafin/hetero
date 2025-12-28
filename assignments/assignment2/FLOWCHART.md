# Flowchart for Assignment 2 Code

```mermaid
flowchart TD
    Start([Start]) --> InitRNG[Initialize RNG srand]
    InitRNG --> PrintStart[Print Start Method]
    PrintStart --> Task1[Task 1: Print Theoretical Info]
    Task1 --> RunTask2[[Call runTask2]]
    RunTask2 --> RunTask3[[Call runTask3]]
    RunTask3 --> RunTask4[[Call runTask4]]
    RunTask4 --> End([End])

    subgraph Task2["Task 2: OpenMP Min/Max"]
    RunTask2_Start(Start runTask2) --> GenArray2[Generate Random Array 10,000]
    GenArray2 --> MeasSeq2[Measure Sequential Min/Max]
    MeasSeq2 --> MeasPar2[Measure Parallel Min/Max OpenMP]
    MeasPar2 --> PrintRes2[Print Results]
    PrintRes2 --> Compare[Compare Sequential & Parallel Results]
    Compare -- Match --> PrintSpeedup[Print Speedup]
    Compare -- Mismatch --> PrintError[Print Error]
    PrintSpeedup --> RunTask2_End(Return)
    PrintError --> RunTask2_End
    end

    subgraph Task3["Task 3: OpenMP Selection Sort"]
    RunTask3_Start(Start runTask3) --> LoopSizes3{For each size in 1000, 10000}
    LoopSizes3 --> GenArray3[Generate Random Array]
    GenArray3 --> CopyArray3[Create Copy for Parallel]
    CopyArray3 --> MeasSeq3[Measure Sequential Selection Sort]
    MeasSeq3 --> MeasPar3[Measure Parallel Selection Sort OpenMP]
    MeasPar3 --> PrintRes3[Print Times & Speedup]
    PrintRes3 --> LoopSizes3
    LoopSizes3 -- Done --> RunTask3_End(Return)
    end

    subgraph Task4["Task 4: CUDA Merge Sort"]
    RunTask4_Start(Start runTask4) --> CheckCUDA{Is CUDA Available?}
    CheckCUDA -- Yes --> LoopSizes4{For each size in 10000, 100000}
    LoopSizes4 --> GenArray4[Generate Random Array]
    GenArray4 --> MeasCUDA[Measure mergeSortCUDA]
    MeasCUDA --> AllocGPU[Allocate GPU Memory]
    AllocGPU --> CopyH2D[Copy Host to Device]
    CopyH2D --> SortLoop{Merge Sort Loop width *= 2}
    SortLoop --> LaunchKernel[Launch mergeKernel]
    LaunchKernel --> Sync[cudaDeviceSynchronize]
    Sync --> SortLoop
    SortLoop -- Done --> CopyD2H[Copy Device to Host]
    CopyD2H --> FreeGPU[Free GPU Memory]
    FreeGPU --> PrintTime4[Print Execution Time]
    PrintTime4 --> LoopSizes4
    LoopSizes4 -- Done --> RunTask4_End(Return)
    CheckCUDA -- No --> PrintSkip[Print Skipping Task 4]
    PrintSkip --> RunTask4_End
    end
