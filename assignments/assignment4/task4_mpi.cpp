#include <mpi.h>      // Include MPI library header
#include <iostream>   // Include IO stream library for output
#include <vector>     // Include vector container
#include <numeric>    // Include numeric library for std::iota
#include <cmath>      // Include cmath for math functions if needed

// Function to perform computation on the data
// Takes a reference to a vector of integers to modify in place
void process_data(std::vector<int>& data) {
    // Loop through each element in the data vector
    for (size_t i = 0; i < data.size(); ++i) {
        // Multiply the current element by 2 (simple computation)
        data[i] = data[i] * 2;
    }
}

// Main function, entry point of the program
int main(int argc, char** argv) {
    // Initialize the MPI environment
    // Passes command line arguments to MPI implementation
    MPI_Init(&argc, &argv);

    // Variable to store the total number of processes
    int world_size;
    // Get the number of processes in the communicator
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Variable to store the rank (ID) of the current process
    int world_rank;
    // Get the rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Define the total size of the array to process
    const int N = 10000000; // 10 Million elements

    // Check if the array size is divisible by the number of processes
    // This simplifies the logic for this assignment
    if (N % world_size != 0) {
        // If not divisible, only the root process prints an error
        if (world_rank == 0) {
            std::cerr << "Error: N (" << N << ") must be divisible by world_size (" << world_size << ")" << std::endl;
        }
        // Finalize MPI before exiting with error
        MPI_Finalize();
        return 1;
    }

    // Calculate the number of elements each process handles
    int n_local = N / world_size;
    // Allocate memory for the local chunk of data
    std::vector<int> local_data(n_local);
    // Vector for the full dataset (only needed on root)
    std::vector<int> global_data;

    // Only the root process (Rank 0) initializes the global data
    if (world_rank == 0) {
        // Resize vector to hold all N elements
        global_data.resize(N);
        // Fill the vector with sequential numbers starting from 1
        std::iota(global_data.begin(), global_data.end(), 1); 
        // Print information about the run
        std::cout << "Starting Task 4 with " << world_size << " processes." << std::endl;
        std::cout << "Array size: " << N << " elements." << std::endl;
    }

    // Synchronize all processes before starting the timer
    // Ensures accurate timing of the parallel section
    MPI_Barrier(MPI_COMM_WORLD);
    // Start the MPI wall clock timer
    double start_time = MPI_Wtime();

    // 1. Scatter data from root to all processes
    // global_data.data(): send buffer (significant only at root)
    // n_local: number of elements to send to each process
    // MPI_INT: data type of sent elements
    // local_data.data(): receive buffer
    // n_local: number of elements to receive
    // MPI_INT: data type of received elements
    // 0: root rank
    // MPI_COMM_WORLD: communicator
    MPI_Scatter(global_data.data(), n_local, MPI_INT, 
                local_data.data(), n_local, MPI_INT, 
                0, MPI_COMM_WORLD);

    // 2. Perform local computation on the received chunk
    process_data(local_data);

    // 3. Gather results back to root
    // local_data.data(): send buffer (local results)
    // n_local: number of elements to send
    // MPI_INT: data type
    // global_data.data(): receive buffer (significant only at root)
    // n_local: number of elements to receive from *each* process
    // MPI_INT: data type
    // 0: root rank
    // MPI_COMM_WORLD: communicator
    MPI_Gather(local_data.data(), n_local, MPI_INT, 
               global_data.data(), n_local, MPI_INT, 
               0, MPI_COMM_WORLD);

    // Stop the timer
    double end_time = MPI_Wtime();
    // Calculate the duration for this specific process
    double local_duration = end_time - start_time;

    // To get the total parallel execution time, we take the maximum duration across all processes
    // This accounts for the slowest process
    double max_duration;
    // Reduce the local_duration values to find the maximum
    // &local_duration: input buffer
    // &max_duration: output buffer (significant only at root)
    // 1: count
    // MPI_DOUBLE: data type
    // MPI_MAX: operation (maximum)
    // 0: root rank
    // MPI_COMM_WORLD: communicator
    MPI_Reduce(&local_duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Only the root process prints the final results
    if (world_rank == 0) {
        // Print the total execution time in milliseconds
        std::cout << "Total Execution Time: " << max_duration * 1000.0 << " ms" << std::endl;

        // Verification step (Check a few elements to ensure correctness)
        bool passed = true;
        // Check the first element (should be 1 * 2 = 2)
        if (global_data[0] != 2) passed = false;
        // Check the last element (should be N * 2)
        if (global_data[N-1] != N * 2) passed = false;
        
        // Output the verification result
        if (passed) {
             std::cout << "Verification: PASSED (First: " << global_data[0] << ", Last: " << global_data[N-1] << ")" << std::endl;
        } else {
             std::cout << "Verification: FAILED" << std::endl;
        }
    }

    // Finalize the MPI environment before exiting
    MPI_Finalize();
    return 0;
}
