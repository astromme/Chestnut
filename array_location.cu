__global__ void function_name(int width, int height, int depth, vars...)
{
    int index = threadIdx.x + 
                threadIdx.y * blockDim.x +
                threadIdx.z * blockDim.x * blockDim.y +
                blockIdx.x  * blockDim.x * blockDim.y * blockDim.z +
                blockIdx.y  * blockDim.x * blockDim.y * blockDim.z * gridDim.x +
                blockIdx.z  * blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y;

    // 1d case
    int x = index;

    // 2d case
    int x = index % width;
    int y = index / width;

    // 3d case
    int x = index % width
    int y = (index / width) % height;
    int z = index / (width * height);

    if ((x > width) || (y > height) || (z > depth)) {
      return;
    }

    result[idx] = calculated_value;
}

