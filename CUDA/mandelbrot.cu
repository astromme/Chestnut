#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <iostream>

#define WIDTH 800
#define HEIGHT 500
#define ITERATIONS 100

__global__ void display_example(unsigned char *display)
{
    int idx = blockIdx.x + threadIdx.y*WIDTH;

    int _x = blockIdx.x;
    int _y = threadIdx.y;

    int _width = WIDTH;
    int _height = HEIGHT;

    float xleft = -2.5;
    float xright = 1;
    float ytop = 1;
    float ybottom = -1;


    float x0 = ((((float)_x / _width) * (xright - xleft)) + xleft);
    float y0 = ((((float)_y / _height) * (ytop - ybottom)) + ybottom);

    float x;
    x = 0;
    float y;
    y = 0;

    int iteration = 0;
    int max_iteration = 1000;

    while (((((x * x) + (y * y)) <= (2 * 2)) && (iteration < max_iteration))) {
        float xtemp;
        xtemp = (((x * x) - (y * y)) + x0);
        y = ((2 * (x * y)) + y0);
        x = xtemp;
        iteration = (iteration + 1);
    }

    if ((iteration == max_iteration)) {
        iteration = 0;
    }

    display[idx] = iteration;
}


int main(int argc, char** argv) {
  size_t spaceNeeded = WIDTH*HEIGHT*sizeof(unsigned char);

  unsigned char* deviceDisplay;

  cudaMalloc((void**)&deviceDisplay, spaceNeeded);

  //printBoard(hostOriginalState, rows, columns);

  // invoke the kernel
  for (int iteration=0; iteration<ITERATIONS; iteration++) {
    //if (iteration % (ITERATIONS/100) == 0) {
    //  std::cout << "Iteration " << iteration << std::endl;
    //}
    display_example<<<dim3(WIDTH, 1), dim3(1, HEIGHT)>>>(deviceDisplay);
  }

  // retrieve the results from the device
  //debug("copy back", cudaMemcpy(hostFinalState, deviceCurrentBoardState, spaceNeeded, cudaMemcpyDeviceToHost));

  //printBoard(hostFinalState, rows, columns);

  // free up the allocated memory on the device
  cudaFree(deviceDisplay);

  return 0;
}
