#include <stdio.h>
#include <cutil.h>

// IMPORTANT NOTE: for this data size, your graphics card should have at least 512 megabytes of memory.
// If your GPU has less memory, then you will need to decrease this data size.

#define MAX_DATA_SIZE           1*1024*32            // about 32 million elements. 
// The max data size must be an integer multiple of 128*256, because each block will have 256 threads,
// and the block grid width will be 128. These are arbitrary numbers I choose.


void GoldenBrick(float *pA, float *pB, float *pResult, int count)
{
  for (int i=0; i < count; i++)
  {
    //pResult[count] = pA[count] * pB[count];
    //pResult[count] = pA[count] * pB[count] / 12.34567;
    //pResult[count] = sqrt(pA[count] * pB[count] / 12.34567);
    pResult[count] = sqrt(pA[count] * pB[count] / 12.34567) * sin(pA[count]);
  }
}

__global__ void multiplyNumbersGPU(float *pDataA, float *pDataB, float *pResult)
{
  // Because of the simplicity of this tutorial, we are going to assume that
  // every block has 256 threads. Each thread simply multiplies two numbers,
  // and then stores the result.
  
  // The grid of blocks is 128 blocks long.
  
  int tid = (blockIdx.y * 128 * 256) + blockIdx.x * 256 + threadIdx.x;    // This gives every thread a unique ID.
  // By no coincidence, we'll be using this thread ID to determine which data elements to multiply.
  
  //pResult[tid] = pDataA[tid] * pDataB[tid];             // Each thread only multiplies one data element.
  //pResult[tid] = pDataA[tid] * pDataB[tid] / 12.34567;
  //pResult[tid] = sqrt(pDataA[tid] * pDataB[tid] / 12.34567);
  pResult[tid] = sqrt(pDataA[tid] * pDataB[tid] / 12.34567) * sin(pDataA[tid]);
  
}

float* mallocFloats() {
  return (float*) malloc(sizeof(float) * MAX_DATA_SIZE);
}

void mallocDeviceArray(float* deviceArray) {
  CUDA_SAFE_CALL( cudaMalloc((void**)&deviceArray, sizeof(float) * MAX_DATA_SIZE));
}

// main routine that executes on the host
int main(int argc, char* argv[])
{
  cudaSetDevice( 0 );
  //CUT_DEVICE_INIT(argc, argv);
  printf("Initializing data...\n");
  
  float *hostDataA, *hostDataB, *hostResult;
  float *deviceDataA, *deviceDataB, *deviceResult;
  
  hostDataA = mallocFloats();
  hostDataB = mallocFloats();
  hostResult = mallocFloats();
  
  mallocDeviceArray(deviceDataA);
  mallocDeviceArray(deviceDataB);
  mallocDeviceArray(deviceResult);
  
    
  srand(123);
  for(int i = 0; i < MAX_DATA_SIZE; i++)
  {
    hostDataA[i] = (float)rand() / (float)RAND_MAX;
    hostDataB[i] = (float)rand() / (float)RAND_MAX;
  }
  
  int firstRun = 1;       // Indicates if it's the first execution of the for loop
  
  for (int dataAmount = MAX_DATA_SIZE; dataAmount > 128*256; dataAmount /= 2)
  {
    int blockGridWidth = 128;
    int blockGridHeight = (dataAmount / 256) / blockGridWidth;
    
    dim3 blockGridRows(blockGridWidth, blockGridHeight);
    dim3 threadBlockRows(256, 1);
    
    
    // Copy the data to the device
    CUDA_SAFE_CALL( cudaMemcpy(deviceDataA, hostDataA, sizeof(float) * dataAmount, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(deviceDataB, hostDataB, sizeof(float) * dataAmount, cudaMemcpyHostToDevice) );
    
    // Do the multiplication on the GPU
    multiplyNumbersGPU<<<blockGridRows, threadBlockRows>>>(deviceDataA, deviceDataB, deviceResult);
    CUT_CHECK_ERROR("multiplyNumbersGPU() execution failed\n");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    
    // Copy the data back to the host
    CUDA_SAFE_CALL( cudaMemcpy(hostResult, deviceResult, sizeof(float) * dataAmount, cudaMemcpyDeviceToHost) );
    
    if (!firstRun)
    {
    }
    else
    {
      // We discard the results of the first run because of the extra overhead incurred
      // during the first time a kernel is ever executed.
      dataAmount *= 2;        // reset to first run value
    }
  }
  
  printf("Cleaning up...\n");
  CUDA_SAFE_CALL( cudaFree(deviceResult ) );
  CUDA_SAFE_CALL( cudaFree(deviceDataB) );
  CUDA_SAFE_CALL( cudaFree(deviceDataA) );
  free(hostResult);
  free(hostDataB);
  free(hostDataA);
  
  CUT_EXIT(argc, argv);
}