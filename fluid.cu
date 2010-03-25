// Of course needed for printf()
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>


// Prototypes
__global__ void radiate(float* now, float* next, int rows, int cols, int pitch);
int wrap(int value, int maxValue);
void printArray(float* array, int rows, int cols);

// Host function. Runs on CPU
int main(int argc, char** argv) {
  int rows = 3;
  int cols = 8;
  int width = cols*sizeof(float);
  float *host; // Memory on host (cpu) side
  
  // Memory on device (gpu) side
  float *devNow; 
  float *devNext;
  
  // TODO: Quesetion: Is this guarenteed to be the same for both arrays?
  size_t pitch; // Actual number of columns in device array;
  int devWidth; // pitch/sizeof(float)
  
  // Allocate memory on the host that the gpu can access quickly
  cudaMallocHost((void**)&host, rows*width); 
  // the above is more or less the same as:
  //host = (float*) malloc(N*N*sizeof(float));
  
  // Allocate memory on the device that is of the same size as our host array
  cudaMallocPitch((void**)&devNow, &pitch, width, rows);
  cudaMallocPitch((void**)&devNext, &pitch, width, rows);
  devWidth = pitch/sizeof(float);
  
  for(int i=0; i<rows*cols; i++) {
    host[i] = i;
  }
  
  printf("Initial Array\n");
  printArray(host, rows, cols);
  
  // Copy the array host to dev. We must give it how many bytes to copy hence the sizeof()
  cudaMemcpy2D(devNow, pitch, host, width, width, rows, cudaMemcpyHostToDevice);
  
  // run the 'grow' kernel on the device
  // 1 tells it to have one single block
  // dim3(N,N) tells it to have N by N threads in that block.
  // Give the grow kernel the 'dev' array and the number of columns (N)
  radiate<<<1, dim3(rows, cols)>>>(devNow, devNext, rows, cols, devWidth);
  
  // Once the kernel has run, copy back the 'dev' array to the 'host' array
  cudaMemcpy2D(host, width, devNext, pitch, width, rows, cudaMemcpyDeviceToHost);
  
  // Now it should have incrementing numbers
  printf("After first radiation:\n");
  printArray(host, rows, cols);
  
  // No need to recopy the memory, it's already on the device
  float* temp;
  temp = devNow;
  devNow = devNext;
  devNext = temp;
  radiate<<<1, dim3(rows, cols)>>>(devNow, devNext, rows, cols, devWidth);
  cudaMemcpy2D(host, width, devNext, pitch, width, rows, cudaMemcpyDeviceToHost);
  
  // Now each row should have the same value for all columns
  printf("After second radiation:\n");
  printArray(host, rows, cols);  
  
  // free up the allocated memory on the device
  cudaFree(devNow);
  cudaFree(devNext);
  
  return 0;
}

void printArray(float* array, int rows, int cols) {
  for(int row=0; row<rows; row++) {
    for (int col=0; col<cols; col++) {
      // calculate the spot in the 2d array that we want.
      float e = array[(row * cols) + col];
      printf("%.2f\t", e);
    }
    printf("\n");
  }
  printf("\n");
}

static int wrap(int value, int maxValue) {
  while (value < 0) {
    value += maxValue;
  }
  
  return value % maxValue;
}

__global__ void radiate(float* now, float* next, int rows, int cols, int pitch) {
  int row = threadIdx.x;
  int col = threadIdx.y;
  
  float left = now[row*pitch + wrap(col-1, cols)];
  float right = now[row*pitch + wrap(col+1, cols)];
  float above = now[wrap(row-1, rows)*pitch + col];
  float below = now[wrap(row+1, rows)*pitch + col];
  
  float avg = (left + right + above + below)/4;
  next[row*pitch + col] = avg;
}