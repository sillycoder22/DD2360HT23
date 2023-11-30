#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DataType float

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

// Function to initialize a matrix with random values
void initializeMatrix(DataType *matrix, int numRows, int numColumns) {
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numColumns; ++j) {
            matrix[i * numColumns + j] = (DataType)rand() / RAND_MAX;
        }
    }
}

// Function to compare two matrices
bool compareMatrices(DataType *mat1, DataType *mat2, int numRows, int numColumns) {
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numColumns; ++j) {
            if (fabs(mat1[i * numColumns + j] - mat2[i * numColumns + j]) > 1e-3f) {
                return false; // Matrices are not equal
            }
        }
    }
    return true; // Matrices are equal
}

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numARows && col < numBColumns) {
        DataType value = 0.0;
        for (int i = 0; i < numAColumns; ++i) {
            value += A[row * numAColumns + i] * B[i * numBColumns + col];
        }
        C[row * numBColumns + col] = value;
    }
}

int main(int argc, char **argv) {
    DataType *hostA; // The A matrix
    DataType *hostB; // The B matrix
    DataType *hostC; // The output C matrix
    DataType *resultRef; // The reference result
    DataType *deviceA;
    DataType *deviceB;
    DataType *deviceC;
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;
    int numCColumns;

    //@@ Insert code below to read in numARows, numAColumns, numBRows, numBColumns from args
    if (argc != 5) {
        fprintf(stderr, "Usage: %s numARows numAColumns numBRows numBColumns\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    numARows = atoi(argv[1]);
    numAColumns = atoi(argv[2]);
    numBRows = atoi(argv[3]);
    numBColumns = atoi(argv[4]);

    numCRows = numARows;
    numCColumns = numBColumns;

    printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    //@@ Insert code below to allocate Host memory for input and output
    hostA = (DataType *)malloc(numARows * numAColumns * sizeof(DataType));
    hostB = (DataType *)malloc(numBRows * numBColumns * sizeof(DataType));
    hostC = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));
    resultRef = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));

    //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
    initializeMatrix(hostA, numARows, numAColumns);
    initializeMatrix(hostB, numBRows, numBColumns);
    // Perform matrix multiplication on the CPU for reference
    for (int i = 0; i < numARows; ++i) {
        for (int j = 0; j < numBColumns; ++j) {
            DataType value = 0.0;
            for (int k = 0; k < numAColumns; ++k) {
                value += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
            }
            resultRef[i * numBColumns + j] = value;
        }
    }

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(DataType));
    cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(DataType));
    cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(DataType));

    //@@ Insert code to below to Copy memory to the GPU here
    double start, stop;
    start = cpuSecond();
    cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);
    stop = cpuSecond();
    printf("Time taken for copy from host to device: %f s\n", stop - start);

    //@@ Initialize the grid and block dimensions here
    dim3 blockSize(16, 16); // You can adjust the block size as needed
    dim3 gridSize((numCColumns + blockSize.x - 1) / blockSize.x, (numCRows + blockSize.y - 1) / blockSize.y);

    //@@ Launch the GPU Kernel here
    start = cpuSecond();
    gemm<<<gridSize, blockSize>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
    cudaDeviceSynchronize(); // Ensure the kernel has completed
    stop = cpuSecond();
    printf("Time taken for CUDA kernel execution: %f s\n", stop - start);

    //@@ Copy the GPU memory back to the CPU here
    start = cpuSecond();
    cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);
    stop = cpuSecond();
    printf("Time taken for copy from device to host: %f s\n", stop - start);

    //@@ Insert code below to compare the output with the reference
    if (compareMatrices(hostC, resultRef, numCRows, numCColumns)) {
        printf("Result verification passed!\n");
    } else {
        printf("Result verification failed!\n");
    }

    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    //@@ Free the CPU memory here
    free(hostA);
    free(hostB);
    free(hostC);
    free(resultRef);

    return 0;
}

