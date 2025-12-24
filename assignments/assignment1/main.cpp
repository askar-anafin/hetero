#include <iostream> // Include the iostream library for input and output operations
#include <vector>   // Include the vector library (though we are using raw pointers for this assignment)
#include <cstdlib>  // Include cstdlib for random number generation functions like rand() and srand()
#include <ctime>    // Include ctime to seed the random number generator with the current time
#include <chrono>   // Include chrono for high-resolution timing of code execution
#include <omp.h>    // Include omp.h for OpenMP parallel processing functionality
#include <limits>   // Include limits to get the minimum and maximum representable values for types
#include <iomanip>  // Include iomanip for output formatting (not strictly used but good practice)

using namespace std; // Use the standard namespace to avoid typing std:: repeatedly

// Helper function to fill an array with random numbers within a specified range
void fillArray(int* arr, int size, int min_val, int max_val) {
    for (int i = 0; i < size; ++i) { // Loop through each index of the array from 0 to size-1
        // Generate a random number between min_val and max_val and assign it to the array element
        arr[i] = min_val + rand() % (max_val - min_val + 1); 
    }
}

// Task 1: Dynamically allocate memory for 50,000 integers, fill with random values 1-100, calculate average, and free memory
void task1() {
    cout << "=== Task 1 ===" << endl; // Print the header for Task 1
    int size = 50000; // Define the size of the array as 50,000
    int* arr = new int[size]; // Dynamically allocate memory for an array of 50,000 integers using 'new'

    fillArray(arr, size, 1, 100); // Call the helper function to fill the array with random numbers from 1 to 100

    long long sum = 0; // Initialize a variable 'sum' to 0 to store the total of the array elements. Use long long to prevent overflow
    for (int i = 0; i < size; ++i) { // Loop through each element of the array
        sum += arr[i]; // Add the current element's value to the sum
    }

    double average = static_cast<double>(sum) / size; // Calculate the average by casting sum to double and dividing by size
    cout << "Array size: " << size << endl; // Print the size of the array
    cout << "Average value: " << average << endl; // Print the calculated average value

    delete[] arr; // Free the dynamically allocated memory to prevent memory leaks
    cout << "Memory freed." << endl << endl; // Print a message confirming memory has been freed and print an extra newline
}

// Task 2 & 3: Create an array of 1,000,000 integers and find Min/Max using Sequential vs Parallel algorithms
void task2_and_3() {
    cout << "=== Task 2 & 3 ===" << endl; // Print the header for Task 2 and 3
    int size = 1000000; // Define the size of the array as 1,000,000
    int* arr = new int[size]; // Dynamically allocate memory for the array
    fillArray(arr, size, 0, 100000); // Fill the array with random values between 0 and 100,000

    // Task 2: Sequential Search for Min and Max and measure time
    cout << "[Sequential]" << endl; // Print a label indicating the start of the sequential part
    int min_seq = std::numeric_limits<int>::max(); // Initialize min_seq to the maximum possible integer value
    int max_seq = std::numeric_limits<int>::min(); // Initialize max_seq to the minimum possible integer value

    auto start_seq = chrono::high_resolution_clock::now(); // Record the start time using high_resolution_clock
    for (int i = 0; i < size; ++i) { // Loop through the array sequentially
        if (arr[i] < min_seq) min_seq = arr[i]; // Update min_seq if the current element is smaller
        if (arr[i] > max_seq) max_seq = arr[i]; // Update max_seq if the current element is larger
    }
    auto end_seq = chrono::high_resolution_clock::now(); // Record the end time
    chrono::duration<double> time_seq = end_seq - start_seq; // Calculate the duration of the sequential execution

    cout << "Min: " << min_seq << ", Max: " << max_seq << endl; // Print the found minimum and maximum values
    cout << "Time: " << time_seq.count() << " seconds" << endl; // Print the time taken for the sequential algorithm


    // Task 3: Parallel Search for Min and Max using OpenMP and measure time
    cout << "[Parallel OpenMP]" << endl; // Print a label indicating the start of the parallel part
    int min_par = std::numeric_limits<int>::max(); // Initialize min_par for the parallel version
    int max_par = std::numeric_limits<int>::min(); // Initialize max_par for the parallel version

    auto start_par = chrono::high_resolution_clock::now(); // Record the start time for the parallel execution
    
    // Start an OpenMP parallel region with a for loop. Use reduction to safely update min_par and max_par across threads
    #pragma omp parallel for reduction(min:min_par) reduction(max:max_par)
    for (int i = 0; i < size; ++i) { // Loop through the array; iterations are distributed among threads
        if (arr[i] < min_par) min_par = arr[i]; // Update thread-local min_par if current element is smaller
        if (arr[i] > max_par) max_par = arr[i]; // Update thread-local max_par if current element is larger
    }

    auto end_par = chrono::high_resolution_clock::now(); // Record the end time for the parallel execution
    chrono::duration<double> time_par = end_par - start_par; // Calculate the duration of the parallel execution

    cout << "Min: " << min_par << ", Max: " << max_par << endl; // Print the min and max found by the parallel algorithm
    cout << "Time: " << time_par.count() << " seconds" << endl; // Print the time taken for the parallel algorithm
    // Calculate and print the speedup (Sequential Time / Parallel Time)
    cout << "Speedup: " << time_seq.count() / time_par.count() << "x" << endl; 

    delete[] arr; // Free the allocated memory for the array
    cout << endl; // Print a newline for spacing
}

// Task 4: Create an array of 5,000,000 numbers and calculate average using Sequential vs Parallel OpenMP with reduction
void task4() {
    cout << "=== Task 4 ===" << endl; // Print the header for Task 4
    int size = 5000000; // Define the size of the array as 5,000,000
    int* arr = new int[size]; // Dynamically allocate memory for the large array
    fillArray(arr, size, 1, 100); // Fill the array with random values from 1 to 100

    // Sequential Calculation of Average
    cout << "[Sequential]" << endl; // Label for sequential part
    long long sum_seq = 0; // Initialize sum variable
    auto start_seq = chrono::high_resolution_clock::now(); // Record start time
    
    for (int i = 0; i < size; ++i) { // Loop through the array sequentially
        sum_seq += arr[i]; // Add each element to sum_seq
    }
    double avg_seq = static_cast<double>(sum_seq) / size; // Calculate average
    
    auto end_seq = chrono::high_resolution_clock::now(); // Record end time
    chrono::duration<double> time_seq = end_seq - start_seq; // Calculate elapsed time
    
    cout << "Average: " << avg_seq << endl; // Print sequential average
    cout << "Time: " << time_seq.count() << " seconds" << endl; // Print sequential time

    // Parallel Calculation of Average using OpenMP
    cout << "[Parallel OpenMP]" << endl; // Label for parallel part
    long long sum_par = 0; // Initialize parallel sum variable
    auto start_par = chrono::high_resolution_clock::now(); // Record start time for parallel execution

    // Parallel for loop with reduction on the 'sum_par' variable to safely sum across threads
    #pragma omp parallel for reduction(+:sum_par)
    for (int i = 0; i < size; ++i) { // Loop through the array
        sum_par += arr[i]; // Add element to thread-local sum; reduced at the end
    }
    double avg_par = static_cast<double>(sum_par) / size; // Calculate parallel average

    auto end_par = chrono::high_resolution_clock::now(); // Record end time
    chrono::duration<double> time_par = end_par - start_par; // Calculate elapsed time

    cout << "Average: " << avg_par << endl; // Print parallel average
    cout << "Time: " << time_par.count() << " seconds" << endl; // Print parallel time
    cout << "Speedup: " << time_seq.count() / time_par.count() << "x" << endl; // Print speedup factor

    delete[] arr; // Free the memory allocated for the large array
    cout << endl; // Print a newline
}

// Main function: Entry point of the program
int main() {
    srand(static_cast<unsigned int>(time(0))); // Seed the random number generator so we get different random numbers each run

    task1(); // Execute Task 1
    task2_and_3(); // Execute Task 2 and 3
    task4(); // Execute Task 4

    return 0; // Return 0 to indicate successful program execution
}
