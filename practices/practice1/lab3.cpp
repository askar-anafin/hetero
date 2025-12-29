#include <iostream> // Include the iostream library for input and output operations
#include <cstdlib> // Include the cstdlib library for general purpose functions like rand and malloc
#include <ctime> // Include the ctime library to access time-related functions
#include <omp.h> // Include the OpenMP library header for parallel programming

// Function to calculate average sequentially
double calculate_average_sequential(int* arr, int size) { // Define a function that takes an integer pointer and size, returning a double
    long long sum = 0; // Initialize a variable 'sum' of type long long to 0 to store the total sum
    for (int i = 0; i < size; ++i) { // Loop from 0 to size - 1
        sum += arr[i]; // Add the value at index i to the sum
    }
    return static_cast<double>(sum) / size; // Return the average by casting sum to double and dividing by size
}

int main() { // Main function where the program execution begins
    // Seed random number generator
    std::srand(std::time(0)); // Seed the random number generator with the current time

    // 1. Create dynamic array using pointers
    int size = 10000000; // Define the size of the array as 10,000,000
    int* arr = new int[size]; // Dynamically allocate an array of integers of the specified size

    if (!arr) { // Check if memory allocation failed (pointer is null)
        std::cerr << "Memory allocation failed!" << std::endl; // Print error message to standard error
        return 1; // Return 1 to indicate error
    }

    // Fill with random numbers
    for (int i = 0; i < size; ++i) { // Loop through each element of the array
        arr[i] = std::rand() % 100; // Assign a random number between 0 and 99 to the current element
    }

    std::cout << "Array size: " << size << std::endl; // Print the size of the array to standard output

    // 2. Sequential calculation
    double start_time = omp_get_wtime(); // Get the current time before starting the sequential calculation
    double avg_seq = calculate_average_sequential(arr, size); // Calculate the average sequentially
    double end_time = omp_get_wtime(); // Get the current time after finishing the sequential calculation
    std::cout << "Sequential Average: " << avg_seq << std::endl; // Print the sequential average result
    std::cout << "Sequential Time: " << (end_time - start_time) << " seconds" << std::endl; // Print the time taken for sequential calculation

    // 3. Parallel calculation using OpenMP
    long long sum_parallel = 0; // Initialize a variable 'sum_parallel' to 0 for the parallel sum
    start_time = omp_get_wtime(); // Get the current time before starting the parallel calculation
    
    #pragma omp parallel for reduction(+:sum_parallel) // Parallelize the loop using OpenMP with reduction for sum_parallel
    for (int i = 0; i < size; ++i) { // Loop through each element of the array
        sum_parallel += arr[i]; // Add the current element to the parallel sum
    }
    
    double avg_parallel = static_cast<double>(sum_parallel) / size; // Calculate the parallel average
    end_time = omp_get_wtime(); // Get the current time after finishing the parallel calculation

    std::cout << "Parallel Average: " << avg_parallel << std::endl; // Print the parallel average result
    std::cout << "Parallel Time: " << (end_time - start_time) << " seconds" << std::endl; // Print the time taken for parallel calculation

    if (abs(avg_seq - avg_parallel) < 1e-9) { // Check if the difference between sequential and parallel averages is negligible
        std::cout << "Verification: SUCCESS (Averages match)" << std::endl; // Print success message if they match
    } else {
        std::cout << "Verification: FAILED (Averages do not match)" << std::endl; // Print failure message if they don't match
    }

    // 4. Free memory
    delete[] arr; // Deallocate the memory used by the dynamic array

    return 0; // Return 0 to indicate successful program execution
}
