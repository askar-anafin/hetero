# Flowchart for Assignment 1 Code

```mermaid
graph TD
    Start([Start Program]) --> SeedRand[Seed Random Number Generator]
    SeedRand --> CallTask1[Call task1]

    %% Task 1 Logic
    subgraph Task_1 [Task 1: Dynamic Allocation & Average]
        direction TB
        T1_Start(Start task1) --> T1_Alloc[Allocate int array<br/>Size: 50,000]
        T1_Alloc --> T1_Fill[Fill Array<br/>Random 1-100]
        T1_Fill --> T1_Sum[Iterate and Calculate Sum]
        T1_Sum --> T1_Avg[Calculate Average]
        T1_Avg --> T1_Print[Print Size & Average]
        T1_Print --> T1_Free[Free Memory]
        T1_Free --> T1_End(End task1)
    end
    CallTask1 --> T1_Start
    T1_End --> CallTask23[Call task2_and_3]

    %% Task 2 & 3 Logic
    subgraph Task_2_3 [Task 2 & 3: Min/Max Sequential vs Parallel]
        direction TB
        T23_Start(Start task2_and_3) --> T23_Alloc[Allocate int array<br/>Size: 1,000,000]
        T23_Alloc --> T23_Fill[Fill Array<br/>Random 0-100,000]
        
        T23_Fill --> T2_Lbl[<b>Sequential Search</b>]
        T2_Lbl --> T2_TimeStart[Start Timer]
        T2_TimeStart --> T2_Loop[Loop and Find Min/Max]
        T2_Loop --> T2_TimeEnd[Stop Timer]
        T2_TimeEnd --> T2_Print[Print Result & Time]

        T2_Print --> T3_Lbl[<b>Parallel Search (OpenMP)</b>]
        T3_Lbl --> T3_TimeStart[Start Timer]
        T3_TimeStart --> T3_Loop[Parallel Loop with Reduction]
        T3_Loop --> T3_TimeEnd[Stop Timer]
        T3_TimeEnd --> T3_Print[Print Result & Time]
        
        T3_Print --> T23_Speedup[Calculate & Print Speedup]
        T23_Speedup --> T23_Free[Free Memory]
        T23_Free --> T23_End(End task2_and_3)
    end
    CallTask23 --> T23_Start
    T23_End --> CallTask4[Call task4]

    %% Task 4 Logic
    subgraph Task_4 [Task 4: Large Array Average Sequential vs Parallel]
        direction TB
        T4_Start(Start task4) --> T4_Alloc[Allocate int array<br/>Size: 5,000,000]
        T4_Alloc --> T4_Fill[Fill Array<br/>Random 1-100]

        T4_Fill --> T4_SeqLbl[<b>Sequential Calc</b>]
        T4_SeqLbl --> T4_S_Start[Start Timer]
        T4_S_Start --> T4_S_Loop[Loop and Accumulate Sum]
        T4_S_Loop --> T4_S_End[Stop Timer]
        T4_S_End --> T4_S_Print[Print Average & Time]

        T4_S_Print --> T4_ParLbl[<b>Parallel Calc (OpenMP)</b>]
        T4_ParLbl --> T4_P_Start[Start Timer]
        T4_P_Start --> T4_P_Loop[Parallel Loop with Reduction]
        T4_P_Loop --> T4_P_End[Stop Timer]
        T4_P_End --> T4_P_Print[Print Average & Time]

        T4_P_Print --> T4_Speedup[Calculate & Print Speedup]
        T4_Speedup --> T4_Free[Free Memory]
        T4_Free --> T4_End(End task4)
    end
    CallTask4 --> T4_Start
    T4_End --> End([End Program])
```
