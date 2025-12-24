#include <iostream>     // Include input/output stream library
#include <vector>       // Include vector container library
#include <random>       // Include random number generation library
#include <chrono>       // Include time utilities library
#include <functional>   // Include functional library for std::function
#include <iomanip>      // Include I/O manipulators for formatting
#include <string>       // Include string library
#include "sorters.h"    // Include custom header for sorting algorithms

using namespace std;         // Use standard namespace
using namespace std::chrono; // Use chrono namespace for time measurement

// Type alias for sort functions (pointer to function taking vector<int>&)
using SortFunction = void(*)(vector<int>&); 

// Struct to hold sorting algorithm implementation and name
struct SortAlgo {
    string name;       // Name of the sorting algorithm
    SortFunction func; // Pointer to the sorting function
};

// Function to generate a random array of a given size
vector<int> generateArray(int size) {
    if (size <= 0) return {}; // Return empty vector if size is not positive
    vector<int> arr(size);    // Create a vector of the specified size
    random_device rd;         // Initialize random device
    mt19937 gen(rd());        // Initialize Mersenne Twister generator
    uniform_int_distribution<> dist(1, size * 10); // Define random range
    for (int& x : arr) x = dist(gen); // Fill array with random numbers
    return arr;               // Return the generated array
}

// Function to check if the array is sorted in ascending order
bool checkSorted(const vector<int>& arr) {
    for (size_t i = 0; i < arr.size() - 1; ++i) { // Loop through array
        if (arr[i] > arr[i + 1]) return false;    // Check if element is greater than next
    }
    return true; // Return true if sorted
}

// Function to measure the execution time of a sorting algorithm
double measureTime(SortFunction func, vector<int> arr) { // Pass by value to sort a copy
    auto start = high_resolution_clock::now(); // Record start time
    func(arr);                                 // Execute the sorting function
    auto end = high_resolution_clock::now();   // Record end time
    
    if (!checkSorted(arr)) {                   // Verify if the array is sorted
        cerr << "\n[Error] Sort failed validation!" << endl; // Error message
    }
    
    return duration_cast<duration<double>>(end - start).count(); // Return elapsed time in seconds
}

int main() {
    // Define array sizes to test
    // Note: Larger sizes (e.g., 100k) may take long for O(N^2) algorithms
    vector<int> sizes = {1000, 10000, 25000}; 

    // Define the list of algorithms to test
    vector<SortAlgo> algos = {
        {"Seq Bubble", bubbleSort},            // Sequential Bubble Sort
        {"Par Bubble", bubbleSortParallel},    // Parallel Bubble Sort
        {"Seq Select", selectionSort},         // Sequential Selection Sort
        {"Par Select", selectionSortParallel}, // Parallel Selection Sort
        {"Seq Insert", insertionSort},         // Sequential Insertion Sort
        {"Par Insert", insertionSortParallel}  // Parallel Insertion Sort
    };

    cout << "Performance Comparison (Time in Seconds)\n"; // Print header
    cout << "-------------------------------------------------------------------------------\n"; // Separator
    cout << left << setw(10) << "Size"; // Print Size column
    for (const auto& algo : algos) {    // Loop through algorithms
        cout << setw(12) << algo.name;  // Print algorithm names
    }
    cout << "\n-------------------------------------------------------------------------------\n"; // Separator

    for (int size : sizes) { // Iterate through each size
        cout << setw(10) << size << flush; // Print current size
        // Generate a base array to ensure all algorithms sort the same data
        vector<int> baseArr = generateArray(size); 
        
        for (const auto& algo : algos) { // Iterate through each algorithm
            double time = measureTime(algo.func, baseArr); // Measure time
            cout << setw(12) << fixed << setprecision(4) << time << flush; // Print time
        }
        cout << endl; // Newline for next row
    }
    cout << "-------------------------------------------------------------------------------\n"; // Footer
    cout << "* Par Insert falls back to Sequential version due to dependency constraints.\n"; // Note

    return 0; // Return success
}
