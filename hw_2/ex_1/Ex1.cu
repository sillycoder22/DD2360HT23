#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < len) {
    out[tid] = in1[tid] + in2[tid];
  }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

int main(int argc, char **argv) {
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  // Read inputLength from args
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <inputLength>\n", argv[0]);
    exit(1);
  }
  inputLength = atoi(argv[1]);

  printf("The input length is %d\n", inputLength);

  // Allocate Host memory for input and output
  hostInput1 = (DataType *)malloc(inputLength * sizeof(DataType));
  hostInput2 = (DataType *)malloc(inputLength * sizeof(DataType));
  hostOutput = (DataType *)malloc(inputLength * sizeof(DataType));
  resultRef = (DataType *)malloc(inputLength * sizeof(DataType));

  // Initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  for (int i = 0; i < inputLength; i++) {
    hostInput1[i] = rand() / (DataType)RAND_MAX;
    hostInput2[i] = rand() / (DataType)RAND_MAX;
    resultRef[i] = hostInput1[i] + hostInput2[i];
  }

  // Allocate GPU memory
  cudaMalloc((void **)&deviceInput1, inputLength * sizeof(DataType));
  cudaMalloc((void **)&deviceInput2, inputLength * sizeof(DataType));
  cudaMalloc((void **)&deviceOutput, inputLength * sizeof(DataType));

  // Copy memory to the GPU
  double startCopyHostToDevice = cpuSecond();
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  double endCopyHostToDevice = cpuSecond();
  printf("Host to Device Copy Time: %f seconds\n", endCopyHostToDevice - startCopyHostToDevice);

  // Initialize the 1D grid and block dimensions
  int blockSize = 256;
  int gridSize = (inputLength + blockSize - 1) / blockSize;

  // Launch the GPU Kernel
  double startKernel = cpuSecond();
  vecAdd<<<gridSize, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  double endKernel = cpuSecond();
  printf("GPU Kernel Execution Time: %f seconds\n", endKernel - startKernel);

  // Copy the GPU memory back to the CPU
  double startCopyDeviceToHost = cpuSecond();
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);
  double endCopyDeviceToHost = cpuSecond();
  printf("Device to Host Copy Time: %f seconds\n", endCopyDeviceToHost - startCopyDeviceToHost);

  // Compare the output with the reference
  for (int i = 0; i < inputLength; i++) {
    if (fabs(hostOutput[i] - resultRef[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(1);
    }
  }

  // Free the GPU memory
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  // Free the CPU memory
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);

  return 0;
}

