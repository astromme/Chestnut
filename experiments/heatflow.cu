
// game of life
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


#define HANDLE_ERROR(a, msg) \
  { \
    cudaError_t ret; \
    if ((ret=(a)) != cudaSuccess) { \
      printf(msg); \
      printf("error %d\n", ret); \
    } \
  }

// must be multiples of 16
#define width 1024
#define height 768

__global__ void heatflow(float *stencil, float *data, float c0, float c1) {

  int x = threadIdx.x + blockIdx.x*blockDim.x;  
  int y = threadIdx.y + blockIdx.y*blockDim.y;  

  float west = data[y*width + (x == 0 ? width-1 : x-1)];
  float east = data[y*width + (x+1 == width ? 0 : x+1)];
  float north = data[(y == 0 ? height-1 : y-1)*width + x];
  float south = data[(y+1 == height ? 0 : y+1)*width + x];

  float new_data = c0*data[y*width+x];
  new_data = new_data + c1*(west + east + north + south);

  new_data = min(1.0, new_data);
  data[y*width+x] = max(0.0, new_data);

  if (stencil[y*width+x] == 1) {
    data[y*width+x] = 1;
  } else if (stencil[y*width+x] == 2) {
    data[y*width+x] = 0;
  }
}

int main (int argc, char *argv[])  {

  float *dev_stencil;
  float *dev_heat;

  float c0 = 0.6;
  float c1 = 0.1; // 4 elements make 0.4

  // allocate memory space on GPU
  HANDLE_ERROR(cudaMalloc((void**)&dev_stencil, sizeof(int)*width*height),
      "malloc dev_stencil") ;
  HANDLE_ERROR(cudaMalloc((void**)&dev_heat, sizeof(int)*width*height),
      "malloc dev_heat") ;

  //TODO: Initialize array contents

  dim3 grid(16,16);  // only 512 threads total per grid
  dim3 blocks(width/16, height/16);

  for (int i=0; i<10000; i++) {
    heatflow<<<blocks,grid>>>(dev_stencil, dev_heat, c0, c1);

    // make the host block until the device is finished with foo
    cudaThreadSynchronize();

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
  }

  cudaFree(dev_stencil);
  cudaFree(dev_heat);

  return 0;
}
