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

__global__ void mandelbrot(int *pixels, float left, float right, float top, float bottom) {

  int grid_x = threadIdx.x + blockIdx.x*blockDim.x;  
  int grid_y = threadIdx.y + blockIdx.y*blockDim.y;  

  // Scale coordinates to be in the given window
  float x0 = (grid_x/width) * (right - left) + left;
  float y0 = (grid_y/height) * (top - bottom) + bottom;

  float x = 0;
  float y = 0;
  int iteration = 0;
  int max_iteration = 1000;

  while (((x*x + y*y) <= (2*2)) && (iteration < max_iteration)) {
    float xtemp = (x*x - y*y) + x0;

    y = 2*x*y + y0;
    x = xtemp;
    iteration = iteration + 1;
  }

  if (iteration == max_iteration)
    pixels[grid_y*width+grid_x] = 0;         // Failed to diverge -- in the set
  else
    pixels[grid_y*width+grid_x] = iteration; // diverged in some number of iterations
}

int main (int argc, char *argv[])  {

  int *dev_pixels;
  float left = -2.5;
  float right = 1;
  float top = 1;
  float bottom = -1;

  // allocate memory space on GPU
  HANDLE_ERROR(cudaMalloc((void**)&dev_pixels, sizeof(int)*width*height),
      "malloc dev_pixels") ;

  dim3 grid(16,16);  // only 512 threads total per grid
  dim3 blocks(width/16, height/16);

  for (int i=0; i<10000; i++) {
    mandelbrot<<<blocks,grid>>>(dev_pixels, left, right, top, bottom);

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

  cudaFree(dev_pixels);

  return 0;
}

