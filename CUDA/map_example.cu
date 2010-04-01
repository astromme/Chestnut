// Of course needed for printf()
#include <stdio.h>
// The main cuda include.
#include <cuda.h>
// only needed for KDevelop4 to find the headers/definitions to give
// syntax highlighting and command completion.
#include <cuda_runtime_api.h>

// Prototypes
template <typename TYPE, typename FCN>
//template <typename TYPE>
__global__ void map(TYPE *array, int cols, FCN fcn);

template <typename TYPE>
TYPE plus(TYPE addBy, TYPE val);

template <typename TYPE>
void printArray(TYPE* array, int rows, int cols);

// Host function. Runs on CPU
int main(int argc, char** argv) {
  int N = 3; // Size of our square array;
  float *host; // Memory on host (cpu) side
  float *dev; // Memory on device (gpu) side
  
  // Allocate memory on the host that the gpu can access quickly
  cudaMallocHost((void**)&host, N*N*sizeof(float)); 
  // the above is more or less the same as:
  //host = (float*) malloc(N*N*sizeof(float));
  
  // Allocate memory on the device that is of the same size as our host array
  cudaMalloc((void**)&dev, N*N*sizeof(float));
  
  // Initialize our 2d array to all 0s. 
  for(int i=0; i<N*N; i++) {
    host[i] = 0;
  }
  
  printf("Initial Array\n");
  printArray<float>(host, N, N);
  
  // Copy the array host to dev. We must give it how many bytes to copy hence the sizeof()
  cudaMemcpy(dev, host, N*N*sizeof(float), cudaMemcpyHostToDevice);

  // run the 'map' kernel on the device
  // 1 tells it to have one single block
  // dim3(N,N) tells it to have N by N threads in that block.
  // Give the grow kernel the 'dev' array and the number of columns (N)

  //map<float><<<1, dim3(N,N)>>>(dev, N);
  
  /*********************************
   * THIS MAKES THE PROGRAM FAIL -- CAN'T DO IT, YO
   *********************************/
  // map<float><<<1, dim3(N,N)>>>(dev, N, plus<float>);
  
  // Once the kernel has run, copy back the 'dev' array to the 'host' array
  cudaMemcpy(host, dev, N*N*sizeof(float), cudaMemcpyDeviceToHost);
  
  // Now it should have incrementing numbers
  printf("After 'map' kernel:\n");
  printArray<float>(host, N, N);
  
  // free up the allocated memory on the device
  cudaFree(dev);
  
  return 0;
}

template <typename TYPE, typename FCN> 
//template <typename TYPE> 
__global__ void map(TYPE *array, int cols, FCN fcn) {
  int row = threadIdx.x;
  int col = threadIdx.y;
  
  // set each slot to its index number
  //array[row*cols + col] += 3;
  array[row*cols + col] = fcn(3, array[row*cols + col]);
}

// note, TYPE must be a(n) int, long, float, double, char
template <typename TYPE>
__device__ TYPE plus(TYPE addBy, TYPE val){
  return val + addBy;
}

template <typename TYPE>
void printArray(TYPE* array, int rows, int cols) {
  for(int row=0; row<rows; row++) {
    for (int col=0; col<cols; col++) {
      // calculate the spot in the 2d array that we want.
      TYPE e = array[(row * cols) + col];
      printf("%.1f\t", e);
    }
    printf("\n");
  }
  printf("\n");
}
