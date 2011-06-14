#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <iostream>
#include <sstream>

texture<float, 2, cudaReadModeElementType> input1Texture;

#define width 16
#define height 8

void cudaCheckError(cudaError_t error, char* message)
{
   if(error!=cudaSuccess) { fprintf(stderr,"ERROR: %s : %i\n",message,error); exit(-1); }
}


std::string stringFromInt(int number)
{
   std::stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}

template <typename T>
void printFullArray2D(const thrust::host_vector<T> &vector) {
  for (int y=0; y<height; y++) {
    for (int x=0; x<width; x++) {
      int i = y*width + x;
      std::cout << ((vector[i] == 0) ? "." : stringFromInt(vector[i]));
      std::cout << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}


struct texture_functor : public thrust::unary_function<float, int> {
  __device__
  float operator()(int index) const
  {
    int x = (index % width);
    int y = (index / width);
    return tex2D(input1Texture, x, y);
  }
};

typedef thrust::counting_iterator<int> IndexIterator;
typedef thrust::transform_iterator<texture_functor, IndexIterator> TextureIterator;

int main() {
  IndexIterator index;

  thrust::host_vector<float> host(width*height);
  thrust::device_vector<float> deviceOutput(width*height);
  thrust::device_vector<float> deviceInput(width*height);

  thrust::copy(index, index+width*height, deviceInput.begin());
  host = deviceInput;
  printFullArray2D<float>(host);

  float *input1Data = thrust::raw_pointer_cast(&(deviceInput[0]));
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>(); 
  size_t offset;
  cudaBindTexture2D(&offset, input1Texture, input1Data, channelDesc, width, height, sizeof(float)*width);

  std::cout << offset << std::endl;
  TextureIterator tex = thrust::make_transform_iterator(index, texture_functor());

  thrust::copy(tex, tex+width*height, deviceOutput.begin());
  host = deviceOutput;
  printFullArray2D<float>(host);
}
