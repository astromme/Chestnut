#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <iostream>

#ifndef __device__
#define __device__
#endif

#ifndef __global__
#define __global__
#endif

__device__ __inline__ int index(dim3 gridDim, dim3 blockIdx) {
  return gridDim.x*blockIdx.y + blockIdx.x;
}

__device__ __inline__ int index(int row, int column, int rows, int columns) {
  if (row < 0) {
    row += rows;
  } else if (row >= rows) {
    row -= rows;
  }

  if (column < 0) {
    column += columns;
  } else if (column >= columns) {
    column -= columns;
  }

  return columns*row + column;
}

__device__ int numNeighbors(int* board, int row, int column, int rows, int columns) {
  int neighborcount = 0;

  for (int i=column-1; i<=column+1; i++) {
    for (int j=row-1; j<=row+1; j++) {
      neighborcount += (board[index(j, i, rows, columns)] != 0);//board[index(j, i , rows, columns)];
    }
  }

  // we counted the cell at row,col above, so now we account for it
  neighborcount -= (board[index(row, column, rows, columns)] != 0);
  return neighborcount;
}

__global__ void computeGeneration(int *currentBoardState, int *nextBoardState) {
  // determine where in the thread grid we are
  /*
  int row = blockIdx.y;
  int column = blockIdx.x;
  int rows = gridDim.y;
  int columns = gridDim.x;
  */
  int row = threadIdx.y;
  int column = threadIdx.x;
  int rows = blockDim.y;
  int columns = blockDim.x;
  int currentIndex = index(row, column, rows, columns);

  int state = 0;

  int neighborcount = numNeighbors(currentBoardState, row, column, rows, columns);


  // if cell is live
  if (currentBoardState[currentIndex]){
    // if <= 1 neighbor, cell dies from loneliness
    if (neighborcount <= 1){
      state = 0;
    }
    // if >= 4 neighbors, cell dies from overpopulation
    else if (neighborcount >= 4){
      state = 0;
    } else {
      state = 1;
    }
  // if cell is dead
  } else {
    if (neighborcount == 3){
      state = 1;
    } else {
      state = 0;
    }
  }

  nextBoardState[currentIndex] = state;
}

void swap_pointers(void** a, void ** b) {
  void* temp = *a;
  *a = *b;
  *b = temp;
}

// prints out board in a nice, human readable format
void printBoard(int* board, int rows, int columns){
  for (int r=0; r<rows; r++){
    for (int c=0; c<columns; c++){
      int index = r*columns + c;
      //printf("%c ", board[index] ? '1' : '.');
      printf("%d ", board[index]);
    }
    printf("\n");
  }
  printf("\n");
}

void debug(char* functionName, int returnValue) {
  if (returnValue != 0) {
    std::cout << functionName << ": " << returnValue << std::endl;
  }
}

int main(int argc, char** argv) {
  int rows = 500;
  int columns = 500;
  int iterations = 1000000;
  int percentLivingAtStart = 30;
  size_t spaceNeeded = rows*columns*sizeof(int);

  int* hostOriginalState = (int*)malloc(spaceNeeded);
  int* deviceCurrentBoardState;
  int* deviceNextBoardState;

  cudaMalloc((void**)&deviceCurrentBoardState, spaceNeeded);
  cudaMalloc((void**)&deviceNextBoardState, spaceNeeded);

  for (int row=0; row<rows; row++){
    for (int column=0; column<columns; column++){
      int index = row*columns + column;
//      int y = row-rows/2;
//      int x = column-columns/2;
//      if (sqrt(x*x + y*y) < 10) {
//         hostOriginalState[index] = true;
//      } else {
//        hostOriginalState[index] = false;
//      }
      int randVal = rand() % 100;
      hostOriginalState[index] = (randVal < percentLivingAtStart) ? 1 : 0;
    }
  }

  //printBoard(hostOriginalState, rows, columns);

  // copy the string to the device
  debug("cudaMemcpy", cudaMemcpy(deviceCurrentBoardState, hostOriginalState, spaceNeeded, cudaMemcpyHostToDevice));

  // invoke the kernel
  for (int iteration=0; iteration<iterations; iteration++) {
    if (iteration % (iterations/100) == 0) {
      std::cout << "Iteration " << iteration << std::endl;
    }
    computeGeneration<<<1, dim3(columns, rows)>>>(deviceCurrentBoardState, deviceNextBoardState);
    cudaThreadSynchronize();
    swap_pointers((void**)&deviceCurrentBoardState, (void**)&deviceNextBoardState);
  }

  // retrieve the results from the device
  int* hostFinalState = hostOriginalState;
  debug("copy back", cudaMemcpy(hostFinalState, deviceCurrentBoardState, spaceNeeded, cudaMemcpyDeviceToHost));

  //printBoard(hostFinalState, rows, columns);

  // free up the allocated memory on the device
  cudaFree(deviceCurrentBoardState);
  cudaFree(deviceNextBoardState);
  free(hostOriginalState);

  return 0;
}


