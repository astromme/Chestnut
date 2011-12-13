// game of life
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// must be multiples of 16
#define width 512
#define height 512

int mandelbrot(int grid_x, int grid_y, float left, float right, float top, float bottom) {
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
    return 0;         // Failed to diverge -- in the set
  else
    return iteration; // diverged in some number of iterations
}

int main (int argc, char *argv[])  {

  int *pixels;
  float left = -2.5;
  float right = 1;
  float top = 1;
  float bottom = -1;

  pixels = (int*)malloc(sizeof(int)*width*height);

  for (int i=0; i<10000; i++) {
    for (int x=0; x<width; x++) {
      for (int y=0; y<height; y++) {
        pixels[y*width+x] = mandelbrot(x, y, left, right, top, bottom);
      }
    }
  }

  free(pixels);

  return 0;
}


