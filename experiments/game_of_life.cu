/* game of life
 * lots of command line options:
 * ./gameoflife <number_iterations> <0:CUDA|1:CPU>  \
 *              <1:printing|0:none> [1: blocks only GPU version]
 *
 * ./gameoflife 10000 1 0   # seqential version, 10000 iterations, no output 
 * ./gameoflife 10000 0 1   # GPU fast version, 10000 iterations,  output 
 * ./gameoflife 10000 0 0 1  # GPU slow version, 10000 iterations, no  output 
 *
 * (newhall, 2011)
 */
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

#define M 500
#define N 500


/**************** sequential GOL ************************/
int get_num_neighbors(int matrix[][M], int n, int x, int y) {

  int i,j, i_index, j_index, num_neighbors;

  num_neighbors=0;
  for(i = x-1; i <= x+1; i++) {
    for(j=y-1; j <= y+1; j++) {
      if((i != x) || (j!=y)) {  // don't include my own value
        i_index = i;
        j_index = j;
        if (i < 0) { i_index = n-1; } 
        if (i == n) {i_index = 0; }
        if (j < 0) { j_index = M-1; } 
        if (j == M) {j_index = 0; }
        num_neighbors += matrix[i_index][j_index];
      }
    }
  }
  return num_neighbors;
}

/* 
 * CPU version: one round of game of life using wrap-around neighbors  
 */
void seq_gameoflife(int matrix[][M], int next_matrix[][M], int n) {

  int i,j, num_neighbors;
  for(i = 0; i < n; i++) {
    for(j = 0; j < M; j++) {
      num_neighbors = get_num_neighbors(matrix, n, i, j);
      if(matrix[i][j]) { // I'm alive 
        if(num_neighbors <= 1 || num_neighbors >= 4) {  // die
          next_matrix[i][j] = 0; 
        } else {          // stay alive
          next_matrix[i][j] = 1; 
        }
      } else {     // I'm dead
        if(num_neighbors == 3) {  // live
          next_matrix[i][j] = 1; 
        } else {   // stay dead
          next_matrix[i][j] = 0; 
        }
      }
    }
  }

}


/**************** GPU GOL ************************/
/* 
 * GPU version: does one round of game of life using wrap-around neighbors  
 */
__global__ void  gameoflife(int *curr, int *next, int blocks_only) {

  int i,j, num_neighbors, offset, i_index, j_index, row, col;


  if(blocks_only) { // M*N blocks, 1 thread each
    row = blockIdx.x;  
    col = blockIdx.y;
    num_neighbors=0;
    for(i=row-1; i <= row+1; i++) {
      for(j=col-1; j <= col+1; j++) {

        if((i != row) || (j!=col)) {  // don't include my own value
          i_index = i;
          j_index = j;
          // slow way: 1 thread M*N blocks
          if (i < 0) { i_index = gridDim.x-1; } 
          if (i == gridDim.x) {i_index = 0; }
          if (j < 0) { j_index = gridDim.y-1; } 
          if (j == gridDim.y) {j_index = 0; }
          // slow way: 1 thread M*N blocks
          // gridDim.y is the total number of blocks in
          // the y dimention
          offset = j_index + i_index*gridDim.y;
          num_neighbors += curr[offset];
        }}}
    offset = col+row*gridDim.y;
  } 
  else {   // ((M+15)/16)x((N+15)/16) blocks, 16x16 threads each

    row = threadIdx.y + blockIdx.y*blockDim.y;  
    col = threadIdx.x + blockIdx.x*blockDim.x;  
    num_neighbors=0;
    for(i=row-1; i <= row+1; i++) {
      for(j=col-1; j <= col+1; j++) {
          if((i != row) || (j!=col)) {  // don't include my own value
            i_index = i;
            j_index = j;
            // fast way: 1 block M*N threads
            if (i < 0) { i_index = N-1; } 
            if (i == N) {i_index = 0; }
            if (j < 0) { j_index = M-1; } 
            if (j == M) {j_index = 0; }
            offset = j_index + i_index*M;   
            if((j_index < M) && (i_index < N)) { 
              num_neighbors += curr[offset];
            }
          }
      } // for j
    }
    row = threadIdx.y + blockIdx.y*blockDim.y;
    col = threadIdx.x + blockIdx.x*blockDim.x;
    offset = col + row*M;
  }

  if(row < N && col < M) {
    if(curr[offset]) { // I'm alive 
      if(num_neighbors <= 1 || num_neighbors >= 4) {  // die
        next[offset] = 0; 
      } else {          // stay alive
        next[offset] = 1;   
      }
    } else {   // I'm dead
      if(num_neighbors == 3) {  // live
        next[offset] = 1;   
      } else {   // stay dead
        next[offset] = 0;   
      }
    }
    // crazy test code
    //next[offset] = threadIdx.y;
  }

}


#define PRINTSIZE 20

void printgameboard(int matrix[][M], int n) {

  int i, j;
  int x,y;

  x = n > PRINTSIZE ? PRINTSIZE: n;
  y = M > PRINTSIZE ? PRINTSIZE: M;
  for (i=0; i < x; i++) {
    for (j=0; j < y; j++) { 
      if(matrix[i][j]){ 
        printf("%2d ", matrix[i][j]);
      } else {
        printf(" * ");
      }
    } 
    printf("\n");
  }
}

int matrix[N][M], next_matrix[N][M];

int main (int argc, char *argv[])  {

  int *dev_grid, *dev_next, i,j, val, iters;
  int sequential = 0, blocks_only=0, print_before_after=0;

  if(argc < 3) {
    printf("usage: ./gameoflife num_iterations <1:seq|0:gpu> <1:print|0:noprint> [1:blocks only gpu]\n");
    exit(1);
  }
  iters = atoi(argv[1]);
  if(iters % 2 != 0) {
    printf("Error: number of iterstions, %d, must be even\n", iters);
    exit(1);
  }
  sequential = atoi(argv[2]);
  print_before_after = atoi(argv[3]);
  if(argc > 4) {
    blocks_only = atoi(argv[4]);
  }

#ifdef ndef
  // andrew's initilization code
  for (i=0; i < N; i++) {
     for (j=0; j < M; j++) {
         val = rand() % 100;
         val = (val < 30) ? 1:0; 
         matrix[i][j] = val; 
         //matrix[i][j] = 0;  // for test pattern
     }
  }
#endif

  // mine
  // initialize game board 
  //srand( (unsigned)time(NULL) );
  for (i=0; i < N; i++) {
     for (j=0; j < M; j++) {
         val = rand();
         val = (val > RAND_MAX/4.0) ? 0:1; 
         matrix[i][j] = val; 
         //matrix[i][j] = 0;  // for test pattern
     }
  }

  // alternating test pattern when rest of matrix is 0 
  matrix[4][5] = 1;
  matrix[5][5] = 1;
  matrix[6][5] = 1;
  if(print_before_after) {
    printf("Starting point game board\n");
    printgameboard(matrix, N);
    printf("\n");
  }
  if(sequential) {  // run sequential version
    printf("sequential version  size=%dx%d iters =%d\n\n", M,N, iters);
      for(i=0; i < iters; i++) {
        if(i%2==0) {
          seq_gameoflife(matrix, next_matrix, N);
        } else {
          seq_gameoflife(next_matrix, matrix, N);
        }
      }

  } else {  // run GPU  version
    // allocate memory space on GPU
    HANDLE_ERROR(cudaMalloc((void**)&dev_grid, sizeof(int)*N*M),
        "malloc dev_grid") ;
    HANDLE_ERROR(cudaMalloc((void**)&dev_next, sizeof(int)*N*M),
        "malloc dev_next") ;


    // copy a and b to GPU 
    HANDLE_ERROR (cudaMemcpy(dev_grid, matrix, sizeof(int)*N*M, 
          cudaMemcpyHostToDevice), "copy dev_grid to GPU") ;

    if(blocks_only) {
      printf("GPU slow version (blks only) size=%dx%d iters =%d\n\n", 
          M,N, iters);
      dim3 slowgrid(N,M);  
      for(i=0; i < iters; i++) {
        if(i%2==0) {
          gameoflife<<<slowgrid,1>>>(dev_grid, dev_next,1);
        } else {
          gameoflife<<<slowgrid,1>>>(dev_next, dev_grid,1);
        }
        // TODO: we don't need to call cudaThreadSycnhronize after
        // each step since the kernel call doesn't return until
        // all calling threads have finished it.  right?
        // cudaThreadSynchronize();
      }
    }
    else {
      printf("GPU fast version (blks of thr blks) size=%dx%d iters =%d\n\n", 
          M,N, iters);
      //printf("the fast way %dx%d (MxN): %dx%d blocks of (16,16) threads\n",
      //    M, N, (M+15)/16,  (N+15)/16);
      dim3 threads(16,16);  // only 512 threads total per grid
      dim3 blocks((M+15)/16, (N+15)/16);
      for(i=0; i < iters; i++) {
        if(i%2==0) {
          gameoflife<<<blocks,threads>>>(dev_grid, dev_next,0);
        } else {
          gameoflife<<<blocks,threads>>>(dev_next, dev_grid,0);
        }
      }
    }

    // copy back result from GPU to CPU
    HANDLE_ERROR (cudaMemcpy(matrix, dev_grid, sizeof(int)*N*M, 
          cudaMemcpyDeviceToHost), "copy dev_grid from GPU") ;
    cudaFree(dev_grid);
    cudaFree(dev_next);
  } 
  printf("%d\n", matrix[0][0]);
  if(print_before_after) {
    printf("After %d iterations:\n", iters);
    printgameboard(matrix, N);
  }
  return 0;
}
