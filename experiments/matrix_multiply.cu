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

// Thread block size
#define BLOCK_SIZE 16
// Forward declaration of the device multiplication function
__global__ void Muld(float*, float*, int, int, float*);
// Host multiplication function
// Compute C = A * B
// hA is the height of A
// wA is the width of A
// wB is the width of B
void Mul(const float* A, const float* B, int hA, int wA, int wB,
				float* C)
{
		int size;
		// Load A and B to the device
		float* Ad;
		size = hA * wA * sizeof(float);
		cudaMalloc((void**)&Ad, size);
		cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
		float* Bd;
		size = wA * wB * sizeof(float);
		cudaMalloc((void**)&Bd, size);
		cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);
		// Allocate C on the device
		float* Cd;
		size = hA * wB * sizeof(float);
		cudaMalloc((void**)&Cd, size);
		// Compute the execution configuration assuming
		// the matrix dimensions are multiples of BLOCK_SIZE
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid(wB / dimBlock.x, hA / dimBlock.y);
		// Launch the device computation
		Muld<<<dimGrid, dimBlock>>>(Ad, Bd, wA, wB, Cd);
		// Read C from the device
		cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);
		// Free device memory
		cudaFree(Ad);
		cudaFree(Bd);
		cudaFree(Cd);
}

// Device multiplication function called by Mul()
// Compute C = A * B
// wA is the width of A
// wB is the width of B
__global__ void Muld(float* A, float* B, int wA, int wB, float* C)
{
		// Block index
		int bx = blockIdx.x;
		int by = blockIdx.y;
		// Thread index
		int tx = threadIdx.x;
		int ty = threadIdx.y;
		// Index of the first sub-matrix of A processed by the block
		int aBegin = wA * BLOCK_SIZE * by;
		// Index of the last sub-matrix of A processed by the block
		int aEnd = aBegin + wA - 1;
		// Step size used to iterate through the sub-matrices of A
		int aStep = BLOCK_SIZE;
		// Index of the first sub-matrix of B processed by the block
		int bBegin = BLOCK_SIZE * bx;
		// Step size used to iterate through the sub-matrices of B
		int bStep = BLOCK_SIZE * wB;
		// The element of the block sub-matrix that is computed
		// by the thread
		float Csub = 0;
		// Loop over all the sub-matrices of A and B required to
		// compute the block sub-matrix
		for (int a = aBegin, b = bBegin;
						a <= aEnd;
						a += aStep, b += bStep) {
				// Shared memory for the sub-matrix of A
				__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
				// Shared memory for the sub-matrix of B
				__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
				// Load the matrices from global memory to shared memory;
				// each thread loads one element of each matrix
				As[ty][tx] = A[a + wA * ty + tx];
				Bs[ty][tx] = B[b + wB * ty + tx];
				// Synchronize to make sure the matrices are loaded
				__syncthreads();
				// Multiply the two matrices together;
				// each thread computes one element
				// of the block sub-matrix
				for (int k = 0; k < BLOCK_SIZE; ++k)
						Csub += As[ty][k] * Bs[k][tx];
				// Synchronize to make sure that the preceding
				// computation is done before loading two new
				// sub-matrices of A and B in the next iteration
				__syncthreads();
		}
		// Write the block sub-matrix to global memory;
		// each thread writes one element
		int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
		C[c + wB * ty + tx] = Csub;
}

int main(int argc, char* argv[]) {
		
  float *dev_a;
  float *dev_b;
  float *dev_c;

  /*
  int width_a = 64*BLOCK_SIZE;
  int height_a = 32*BLOCK_SIZE;
  int width_b = height_a;
  int height_b = 48*BLOCK_SIZE;
  int width_c = width_a;
  int height_c = height_b;
  */

  int width_a = 512;
  int height_a = 512;
  int width_b = 512;
  int height_b = 512;
  int width_c = 512;
  int height_c = 512;

  // allocate memory space on GPU
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, sizeof(float)*width_a*height_a),
      "malloc dev_a") ;
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, sizeof(float)*width_b*height_b),
      "malloc dev_b") ;
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(float)*width_c*height_c),
      "malloc dev_c") ;

  // Compute the execution configuration assuming
  // the matrix dimensions are multiples of BLOCK_SIZE
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(width_b / dimBlock.x, height_a / dimBlock.y);
  // Launch the device computation
  for (int i=0; i<100; i++) {
		Muld<<<dimGrid, dimBlock>>>(dev_a, dev_b, width_a, width_b, dev_c);
  }
  
  //Mul(dev_a, dev_b, height_a, width_a, width_b, dev_c);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
}
