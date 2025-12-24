#ifndef SORTERS_H // Include guard start to prevent double inclusion
#define SORTERS_H // Define guard

#include <vector>    // Include vector library for dynamic arrays
#include <algorithm> // Include algorithms library for swap function
#include <omp.h>     // Include OpenMP library for parallel processing
#include <iostream>  // Include iostream for potential I/O

// Sequential Bubble Sort implementation
void bubbleSort(std::vector<int>& arr) { // Function definition
    int n = arr.size();                  // Get current size of vector
    bool swapped;                        // Flag to track if a swap occurred in the pass
    for (int i = 0; i < n - 1; i++) {    // Outer loop for each pass
        swapped = false;                 // Reset swap flag for this pass
        for (int j = 0; j < n - i - 1; j++) { // Inner loop comparing adjacent elements
            if (arr[j] > arr[j + 1]) {        // Check if elements are in wrong order
                std::swap(arr[j], arr[j + 1]); // Swap the elements
                swapped = true;                // Mark that a swap occurred
            }
        }
        if (!swapped) break; // Optimization: Exit if array is already sorted
    }
}

// Parallel Bubble Sort (Odd-Even Transposition Sort)
// Optimized to reduce fork/join overhead by keeping threads alive.
void bubbleSortParallel(std::vector<int>& arr) { // Function definition
    int n = arr.size(); // Get size of vector
    bool swapped = true; // Initialize to true to enter the while loop

    #pragma omp parallel // Start parallel region
    {
        while (true) { // Loop until the entire array is sorted
            // Check exit condition determined in previous iteration
            #pragma omp barrier // Synchronize all threads
            if (!swapped) break; // Exit loop if no swaps occurred previously
            
            // Prepare for the current iteration
            #pragma omp barrier // Ensure all threads see 'break' check before resetting
            #pragma omp single  // Execute block by a single thread
            swapped = false;    // Reset global swap flag
            // Implicit barrier exists at end of 'omp single'
            
            // Odd phase
            // Use local variable to reduce cache contention on 'swapped'
            bool local_swapped = false; // Thread-local swap flag

            #pragma omp for // Distribute loop iterations across threads
            for (int i = 1; i < n - 1; i += 2) { // Iterate over odd indices
                if (arr[i] > arr[i + 1]) { // Check if out of order
                    std::swap(arr[i], arr[i + 1]); // Swap elements
                    local_swapped = true; // Mark local swap
                }
            }

            // Even phase
            #pragma omp for // Distribute loop iterations across threads
            for (int i = 0; i < n - 1; i += 2) { // Iterate over even indices
                if (arr[i] > arr[i + 1]) { // Check if out of order
                    std::swap(arr[i], arr[i + 1]); // Swap elements
                    local_swapped = true; // Mark local swap
                }
            }

            // Update global swapped flag if needed
            if (local_swapped) { // If this thread performed a swap
                // If any thread swapped, multiple passes might be needed.
                // We use atomic write to safely update the shared variable.
                if (!swapped) { // Optimization check to avoid unnecessary writes
                    #pragma omp atomic write // Atomic operation for thread safety
                    swapped = true; // Set global flag to true
                }
            }
        }
    }
}

// Sequential Selection Sort implementation
void selectionSort(std::vector<int>& arr) { // Function definition
    int n = arr.size(); // Get size of vector
    for (int i = 0; i < n - 1; i++) { // Loop through each element
        int min_idx = i; // Assume current element is minimum
        for (int j = i + 1; j < n; j++) { // Scan remaining unsorted part
            if (arr[j] < arr[min_idx]) // Check for smaller element
                min_idx = j; // Update index of new minimum
        }
        std::swap(arr[i], arr[min_idx]); // Swap current element with found minimum
    }
}

// Parallel Selection Sort implementation
// Optimized to keep parallel region outer to reduce overhead.
void selectionSortParallel(std::vector<int>& arr) { // Function definition
    int n = arr.size(); // Get size of vector
    int min_idx;        // Shared variable for index of minimum element
    int min_val;        // Shared variable for value of minimum 

    #pragma omp parallel // Start parallel region
    {
        for (int i = 0; i < n - 1; i++) { // Loop through array positions (sequential flow)
            // Initialize shared min variables for this pass
            #pragma omp single // Executed by one thread
            {
                min_idx = i;      // Set current index as min index
                min_val = arr[i]; // Set current value as min value
            }
            // Implicit barrier ensures all threads wait for initialization

            // Thread-local search for minimum in the remaining part
            int local_min_idx = -1;       // Initialize local min index
            int local_min_val = min_val;  // Initialize with current global min

            #pragma omp for nowait // Distribute loop, no implied barrier at end
            for (int j = i + 1; j < n; j++) { // key loop for finding min
                if (arr[j] < local_min_val) { // Compare against local min
                    local_min_val = arr[j];   // Update local min value
                    local_min_idx = j;        // Update local min index
                }
            }

            // Reduce local minimums to global minimum
            if (local_min_val < min_val) { // Optimization: only attempt update if locally found smaller
                #pragma omp critical // Critical section for safe shared update
                {
                    if (local_min_val < min_val) { // Double check inside lock
                        min_val = local_min_val; // Update global min value
                        min_idx = local_min_idx; // Update global min index
                    }
                }
            }
            #pragma omp barrier // Synchronize threads before swapping

            // Perform swap
            #pragma omp single // Executed by one thread
            std::swap(arr[i], arr[min_idx]); // Swap element at i with found minimum
            // Implicit barrier ensures swap completes before next pass
        }
    }
}

// Sequential Insertion Sort implementation
void insertionSort(std::vector<int>& arr) { // Function definition
    int n = arr.size(); // Get size of vector
    for (int i = 1; i < n; i++) { // Loop from second element
        int key = arr[i]; // Store current element value
        int j = i - 1; // Start comparison with previous element
        while (j >= 0 && arr[j] > key) { // Shift elements greater than key
            arr[j + 1] = arr[j]; // Move element one position ahead
            j = j - 1; // Move to previous position
        }
        arr[j + 1] = key; // Place key in its correct position
    }
}

// Parallel Insertion Sort implementation
// Wrapper for consistency as Insertion Sort is hard to parallelize efficiently.
void insertionSortParallel(std::vector<int>& arr) { // Function definition
    // Insertion sort contains loop dependencies preventing simple parallelization
    insertionSort(arr); // Fallback to sequential implementation
}

#endif // SORTERS_H
