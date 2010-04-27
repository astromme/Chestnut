int main() {
  int* host;
  int* dev;
  int N = 10;

  cudaMallocHost((void*)&host, N*sizeof(int));
  cudaMalloc((void*)&dev, N*sizeof(int));

  for (int i=0; i<N; i++){
    host[i] = 2;
  }

  // Copy the array host to dev
  cudaMemcpy(dev, host, N*sizeof(int),
      cudaMemcpyHostToDevice);

  // run the map kernel on the device
  map<int><<<1, dim3(N)>>>(dev, N);

  // copy back dev array to host array
  cudaMemcpy(host, dev, N*sizeof(int),
      cudaMemcpyDeviceToHost);

  return 0;
}

__global__ void map(int* array, int cols){
  int row = threadIdx.x;
  int col = threadIdx.y;

  array[row*cols + col] += 1;
}                                     
