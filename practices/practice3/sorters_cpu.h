#ifndef SORTERS_CPU_H // Include guard to prevent multiple inclusions
#define SORTERS_CPU_H // Definition of the include guard

#include <vector> // Include vector library for dynamic arrays
#include <algorithm> // Include algorithm library for usage of std::swap and others
#include <iostream> // Include IO stream for input/output operations

// --- Merge Sort ---
template <typename T> // Template definition to allow sorting of any data type
void merge(std::vector<T>& arr, int left, int mid, int right) { // Merge function to combine two sorted subarrays
    int n1 = mid - left + 1; // Calculate size of the left subarray
    int n2 = right - mid; // Calculate size of the right subarray

    std::vector<T> L(n1), R(n2); // Create temporary vectors for left and right subarrays

    for (int i = 0; i < n1; i++) // Loop to copy data to left temporary array
        L[i] = arr[left + i]; // Copy element to L
    for (int j = 0; j < n2; j++) // Loop to copy data to right temporary array
        R[j] = arr[mid + 1 + j]; // Copy element to R

    int i = 0, j = 0, k = left; // Initialize indices: i for L, j for R, k for merged array
    while (i < n1 && j < n2) { // Loop while elements exist in both subarrays
        if (L[i] <= R[j]) { // Compare current elements of L and R
            arr[k] = L[i]; // If L[i] is smaller or equal, place it in the original array
            i++; // Move to next element in L
        } else { // If R[j] is smaller
            arr[k] = R[j]; // Place R[j] in the original array
            j++; // Move to next element in R
        }
        k++; // Move to next position in the original array
    }

    while (i < n1) { // Loop to copy remaining elements of L, if any
        arr[k] = L[i]; // Place remaining element in original array
        i++; // Move to next element in L
        k++; // Move to next position
    }

    while (j < n2) { // Loop to copy remaining elements of R, if any
        arr[k] = R[j]; // Place remaining element in original array
        j++; // Move to next element in R
        k++; // Move to next position
    }
}

template <typename T> // Template definition for generic type
void mergeSort(std::vector<T>& arr, int left, int right) { // Recursive merge sort function
    if (left >= right) return; // Base case: if subarray has 0 or 1 element, it is sorted
    int mid = left + (right - left) / 2; // Calculate the middle index to split the array
    mergeSort(arr, left, mid); // Recursively sort the first half
    mergeSort(arr, mid + 1, right); // Recursively sort the second half
    merge(arr, left, mid, right); // Merge the two sorted halves
}

template <typename T> // Template definition
void cpuMergeSort(std::vector<T>& arr) { // Public wrapper function for merge sort
    if (arr.empty()) return; // Check if array is empty to avoid errors
    mergeSort(arr, 0, arr.size() - 1); // Call the recursive merge sort implementation
}

// --- Quick Sort ---
template <typename T> // Template definition
int partition(std::vector<T>& arr, int low, int high) { // Partition function for Quick Sort
    T pivot = arr[high]; // Choose the last element as the pivot
    int i = (low - 1); // Initialize index of smaller element
    for (int j = low; j <= high - 1; j++) { // Loop from low to high-1
        if (arr[j] < pivot) { // Check if current element is smaller than or equal to pivot
            i++; // Increment index of smaller element
            std::swap(arr[i], arr[j]); // Swap current element with element at index i
        }
    }
    std::swap(arr[i + 1], arr[high]); // Swap pivot with the element at i+1
    return (i + 1); // Return the partitioning index
}

template <typename T> // Template definition
void quickSort(std::vector<T>& arr, int low, int high) { // Recursive Quick Sort function
    if (low < high) { // Check if indices are valid (more than one element)
        int pi = partition(arr, low, high); // Partition the array and get pivot index
        quickSort(arr, low, pi - 1); // Recursively sort elements before partition and pivot
        quickSort(arr, pi + 1, high); // Recursively sort elements after partition and pivot
    }
}

template <typename T> // Template definition
void cpuQuickSort(std::vector<T>& arr) { // Public wrapper function for Quick Sort
    if (arr.empty()) return; // Check if array is empty
    quickSort(arr, 0, arr.size() - 1); // Call the recursive quick sort implementation
}

// --- Heap Sort ---
template <typename T> // Template definition
void heapify(std::vector<T>& arr, int n, int i) { // Heapify function to maintain heap property
    int largest = i; // Initialize largest as root
    int left = 2 * i + 1; // Calculate index of left child
    int right = 2 * i + 2; // Calculate index of right child

    if (left < n && arr[left] > arr[largest]) // Check if left child exists and is greater than largest
        largest = left; // Update largest to left child

    if (right < n && arr[right] > arr[largest]) // Check if right child exists and is greater than largest
        largest = right; // Update largest to right child

    if (largest != i) { // If largest is not root
        std::swap(arr[i], arr[largest]); // Swap root with largest
        heapify(arr, n, largest); // Recursively heapify the affected sub-tree
    }
}

template <typename T> // Template definition
void cpuHeapSort(std::vector<T>& arr) { // Public function for Heap Sort
    int n = arr.size(); // Get the number of elements
    if (n == 0) return; // Return if array is empty

    // Build heap (rearrange array)
    for (int i = n / 2 - 1; i >= 0; i--) // Loop from the last non-leaf node down to root
        heapify(arr, n, i); // Call heapify to build max heap

    // One by one extract an element from heap
    for (int i = n - 1; i > 0; i--) { // Loop from end of array down to 1
        std::swap(arr[0], arr[i]); // Move current root (max element) to end
        heapify(arr, i, 0); // call max heapify on the reduced heap
    }
}

#endif // SORTERS_CPU_H // End of include guard
