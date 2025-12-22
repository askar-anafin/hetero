#ifndef SORTERS_H
#define SORTERS_H

#include <vector>
#include <algorithm>
#include <omp.h>
#include <iostream>

// Sequential Bubble Sort
void bubbleSort(std::vector<int>& arr) {
    int n = arr.size();
    bool swapped;
    for (int i = 0; i < n - 1; i++) {
        swapped = false;
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}

// Parallel Bubble Sort (Odd-Even Transposition Sort)
// Optimized to reduce fork/join overhead by keeping threads alive.
void bubbleSortParallel(std::vector<int>& arr) {
    int n = arr.size();
    bool swapped = true;

    #pragma omp parallel
    {
        while (true) {
            // Check exit condition from previous iteration
            #pragma omp barrier
            if (!swapped) break;
            
            // Result for this iteration
            #pragma omp barrier
            #pragma omp single
            swapped = false;
            // Implicit barrier at end of single
            
            // Odd phase
            // Use logical swapped to avoid contention on shared 'swapped'
            bool local_swapped = false;

            #pragma omp for
            for (int i = 1; i < n - 1; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    std::swap(arr[i], arr[i + 1]);
                    local_swapped = true;
                }
            }

            // Even phase
            #pragma omp for
            for (int i = 0; i < n - 1; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    std::swap(arr[i], arr[i + 1]);
                    local_swapped = true;
                }
            }

            if (local_swapped) {
                // If any thread swapped, we need another pass.
                // Relaxed consistency is fine as long as we eventually see it.
                // Atomic write ensures visibility.
                if (!swapped) { // optimization to avoid excessive writes
                    #pragma omp atomic write
                    swapped = true;
                }
            }
        }
    }
}

// Sequential Selection Sort
void selectionSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_idx])
                min_idx = j;
        }
        std::swap(arr[i], arr[min_idx]);
    }
}

// Parallel Selection Sort
// Optimized to keep parallel region outer.
void selectionSortParallel(std::vector<int>& arr) {
    int n = arr.size();
    int min_idx;
    int min_val;

    #pragma omp parallel
    {
        for (int i = 0; i < n - 1; i++) {
            // Initialize shared min for this pass
            #pragma omp single
            {
                min_idx = i;
                min_val = arr[i];
            }
            // Implicit barrier - all threads wait for init

            // Thread-local search
            int local_min_idx = -1;
            int local_min_val = min_val;

            #pragma omp for nowait
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < local_min_val) {
                    local_min_val = arr[j];
                    local_min_idx = j;
                }
            }

            // Reduce to global min
            // Only update if we found something smaller than current global min_val
            // Note: Since we initialized local_min_val to min_val, 
            // we only update if we found strictly smaller.
            if (local_min_val < min_val) {
                #pragma omp critical
                {
                    if (local_min_val < min_val) { // Double-check pattern
                        min_val = local_min_val;
                        min_idx = local_min_idx;
                    }
                }
            }
            #pragma omp barrier

            // Swap
            #pragma omp single
            std::swap(arr[i], arr[min_idx]);
            // Implicit barrier - wait for swap before next iter
        }
    }
}

// Sequential Insertion Sort
void insertionSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

// Parallel Insertion Sort
// Wrapper for consistency.
void insertionSortParallel(std::vector<int>& arr) {
    // Insertion sort is inherently serial.
    insertionSort(arr);
}

#endif
