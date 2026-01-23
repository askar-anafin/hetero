
all: task1_sum task2_scan task3_hybrid task4_mpi

task1_sum: task1_sum.cu
	nvcc -o task1_sum task1_sum.cu

task2_scan: task2_scan.cu
	nvcc -o task2_scan task2_scan.cu

task3_hybrid: task3_hybrid.cu
	nvcc -o task3_hybrid task3_hybrid.cu

task4_mpi: task4_mpi.cpp
	nvcc -o task4_mpi task4_mpi.cpp -I"$(MSMPI_INC)\" -L"$(MSMPI_LIB64)\" -lmsmpi

clean:
	del *.exe *.exp *.lib
