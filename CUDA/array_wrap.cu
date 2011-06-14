#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/generate.h>
#include <iostream>
#include <sstream>


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
void printFullArray2D(const thrust::host_vector<T> &vector, int width, int height) {
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

template <typename T>
struct Array2d {
  T *data;
  int width;
  int height;

  Array2d(T *data, int width, int height) {
    this->data = data;
    this->width = width;
    this->height = height;
  }
  
  Array2d(thrust::device_vector<T> &vector, int width, int height) {
    data = thrust::raw_pointer_cast(&(vector[0]));
    this->width = width;
    this->height = height;
  }

  __host__ __device__
  int calculateIndex(int x, int y, int x_offset, int y_offset) const {
    x = ((x + x_offset) + width)  % width;
    y = ((y + y_offset) + height) % height;

    return y*width + x;
  }

  __host__ __device__
  T shiftedData(int x, int y, int x_offset, int y_offset) const {
    return data[calculateIndex(x, y, x_offset, y_offset)];
  }
};
    
template <typename OutputType, typename Input1Type>
struct array_functor : public thrust::unary_function<int, OutputType> {

  Array2d<Input1Type> input1;
  int width;
  int height;

  array_functor(const Array2d<Input1Type> &input1_, int width_, int height_)
    : input1(input1_), width(width_), height(height_) {}

  __host__ __device__
  OutputType operator()(int index) const
  {
    int x = (index % width);
    int y = (index / width);

    return input1.shiftedData(x, y, 0, -1); // top
  }
};

typedef thrust::counting_iterator<int> IndexIterator;
typedef thrust::transform_iterator<array_functor<float, float>, IndexIterator> ArrayFloatIterator;

int main() {
  int width = 10;
  int height = 10;
  IndexIterator index;

  thrust::host_vector<float> host(width*height);
  thrust::device_vector<float> deviceOutput(width*height);
  thrust::device_vector<float> deviceInput(width*height);

  thrust::copy(index, index+width*height, deviceInput.begin());
  host = deviceInput;

  printFullArray2D<float>(host, width, height);

  Array2d<float> input1Data(deviceInput, width, height);

  ArrayFloatIterator left = thrust::make_transform_iterator(index, array_functor<float, float>(input1Data, width, height));

  thrust::copy(left, left+width*height, deviceOutput.begin());
  host = deviceOutput;
  printFullArray2D<float>(host, width, height);
}

