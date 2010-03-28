// Matix
// 
// Addition
// Subtraction
// Multiplication
// Transpose
// Inverse
// 
//     A               B
// | 1 | 2 |     | 1 | 2 | 3 |
// | 3 | 4 |     | 4 | 5 | 6 |
// | 5 | 6 |     | 7 | 8 | 9 |
// 
//     C
// | 0 | 0 |
// | 0 | 0 |
// 
// 
// A = A + A
// A = A - (oldA)
// 
// B = transpose(A)
// 
// C = A*B


#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stddef.h>

typedef struct {
  float *data;
  int rows;
  int cols;
  int width; // rows + padding
  size_t pitch; // width*sizeof(float)
} Matrix;

// Prototypes
__global__ void gpu_matrix_add(Matrix A, float *dataA, Matrix B, float *dataB, Matrix result, float *dataC);
__global__ void gpu_matrix_subtract(Matrix A, Matrix B, Matrix result);
__global__ void gpu_matrix_multiply(Matrix A, Matrix B, Matrix result);
__global__ void gpu_matrix_transpose(Matrix original, Matrix result);
__global__ void gpu_matrix_inverse(Matrix original, Matrix result);
int wrap(int value, int maxValue);
void printMatrix(Matrix *m);
void printArray(float* array, int rows, int cols);

void initHostMatrix(Matrix *m, int rows, int cols) {
  m->rows = rows;
  m->cols = cols;
  m->width = m->rows * sizeof(float);
  m->pitch = m->width;
  cudaMallocHost((void**)&m->data, m->rows*m->pitch);
}

void initDeviceMatrix(Matrix *m, int rows, int cols) {
  m->rows = rows;
  m->cols = cols;
  m->width = m->rows * sizeof(float);
  cudaMallocPitch((void**)&m->data, &m->pitch, m->width, m->rows);
}

// TODO: check errors
void copy(Matrix *dest, Matrix *source, cudaMemcpyKind kind) {
  cudaMemcpy2D(dest->data, dest->pitch, source->data, source->pitch, source->rows, source->cols, kind);
}
  

int main(int argc, char** argv) {
  Matrix hostA, hostB, hostResult;
  Matrix devA, devA2, devB, devResult;
  
  initHostMatrix(&hostA, 8, 3);
  initHostMatrix(&hostB, 8, 3);
  initHostMatrix(&hostResult, 8, 3);
  
  initDeviceMatrix(&devA, 8, 3);
  initDeviceMatrix(&devA2, 8, 3);
  initDeviceMatrix(&devB, 3, 8);
  initDeviceMatrix(&devResult, 3, 3);
 
  printf("Aligned pitches of A, A2, B, Result:\n%d, %d, %d, %d\n\n", devA.pitch, devA2.pitch, devB.pitch, devResult.pitch);
  
  for(int i=0; i<hostA.rows*hostA.cols; i++) {
    hostA.data[i] = i;
  }
  
  printf("Array A\n");
  printMatrix(&hostA);
  
  /* ========Perform A+A=========== */
  copy(&devA, &hostA, cudaMemcpyHostToDevice);
  gpu_matrix_add<<<1, dim3(devA.rows, devA.cols)>>>(devA, devA.data, devA, devA.data, devA2, devA2.data);
  copy(&hostA, &devA2, cudaMemcpyDeviceToHost);
  
  printf("A2 = A+A:\n");
  printMatrix(&hostA);
  
  /* ========Perform A-A=========== */
  gpu_matrix_subtract<<<1, dim3(devA.rows, devA.cols)>>>(devA, devA, devA2);
  copy(&hostA, &devA2, cudaMemcpyDeviceToHost);
  
  printf("A2-A:\n");
  printMatrix(&hostA);
  
  /* ========Perform transpose(A)=========== */
  gpu_matrix_transpose<<<1, dim3(devA.rows, devA.cols)>>>(devA, devB);
  copy(&hostB, &devB, cudaMemcpyDeviceToHost);
  
  printf("B = transpose(A)\n");
  printMatrix(&hostB);
  
  /* ========Perform A*B=========== */
  //gpu_matrix_multiply(devA, devB, devResult, colsA, rowsResult, colsResult, pitchA, pitchB, pitchResult);
  
  cudaFree(devA.data);
  cudaFree(devA2.data);
  cudaFree(devB.data);
  cudaFree(devResult.data);
  
  return 0;
}

void printMatrix(Matrix *m) {
  printf("rows: %d, cols: %d\n", m->rows, m->cols);
  printArray(m->data, m->rows, m->cols);
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

__device__ int wrap(int value, int maxValue) {
  while (value < 0) {
    value += maxValue;
  }
  
  return value % maxValue;
}

__global__ void gpu_matrix_add(Matrix A, Matrix B, Matrix result) {
  int row = threadIdx.x;
  int col = threadIdx.y;
  
  int index = row*A.pitch + col;
  result.data[index] = A.data[index] + B.data[index];
}

__global__ void gpu_matrix_subtract(Matrix A, Matrix B, Matrix result) {
  int row = threadIdx.x;
  int col = threadIdx.y;
  
  int index = row*A.pitch + col;
  result.data[index] = A.data[index] - B.data[index];
}

__global__ void gpu_matrix_multiply(Matrix A, float *dataA, Matrix B, float *dataB, Matrix result, float *dataC) {
  int row = threadIdx.x;
  int col = threadIdx.y;
  
  int innerLength = A.cols;
  float value = 0;
  int indexA, indexB;
  for (int i=0; i< innerLength; i++) {
    indexA = row*A.pitch + i;
    indexB = i*B.pitch + col;
    
    value += dataA[indexA] * dataB[indexB];
  }
  
  int index = row*result.pitch + col;
  dataC[index] = value;
}

__global__ void gpu_matrix_transpose(Matrix original, Matrix result) {
  int row = threadIdx.x;
  int col = threadIdx.y;
  
  int newRow = original.rows - row;
  int newCol = original.cols - col;
  
  int index = row*original.pitch + col;
  int newIndex = newRow*result.pitch + newCol;
  
  result.data[newIndex] = original.data[index];
}

__global__ void gpu_matrix_inverse(Matrix original, Matrix result) {
  //hehe...
}
