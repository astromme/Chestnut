#!/usr/bin/env python

preamble = """\
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <limits.h>
#include <cstdlib>
#include <iostream>
#include <sstream>

template <typename T>
__global__ void copyWrapAroundAreas(T *array, int width, int height);

std::string stringFromInt(int number)
{
   std::stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}

/*
  Simple debugging function to print out a 2d array. Does not know about
  padding, so send it the padded width/height rather than the base width/height
*/
template <typename T>
static void printFullArray2D(const thrust::host_vector<T> &vector, int width, int height) {
  for (int y=0; y<height; y++) {
    for (int x=0; x<width; x++) {
      int i = y*width + x;
      std::cout << ((vector[i] == 0) ? "." : vector[i]) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

/*
   Slightly smarter padding-aware print function. Send it in the width and the
   padding and it will do the dirty work for you.
*/
template <typename T>
static void printArray2D(const thrust::host_vector<T> &vector, int width, int height, int padding) {
  int paddedWidth = width + 2*padding;
  int paddedHeight = height + 2*padding;

  for (int y=padding; y<height+padding; y++) {
    for (int x=padding; x<width+padding; x++) {
      int i = y*paddedWidth + x;
      std::cout << ((vector[i] == 0) ? "." : stringFromInt(vector[i])) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}


/*
   This is in a struct rather than a namespace because of some C++ typing
   issues that seem to be unresolvable. Consider it a namespace from the usage
   point of view. But otherwise it contains all of the helper functions for
   working with thrust datatypes and for doing wrapping and creation of (very)
   fancy iterators
*/
template <typename T>
struct Chestnut
{
  // our 2d array type mushed into a 1d array
  typedef typename thrust::device_vector<T> vector;

  // Iterators for the above data type
  typedef typename vector::iterator dataIter;
  typedef typename vector::iterator resultIter;

  // constant iterator information for the location in the array + the bounds
  // from these we can compute the x, y, width, height inside of the kernel
  typedef typename thrust::counting_iterator<int> indexIterType;

  typedef typename thrust::constant_iterator<int> widthIterType;
  typedef typename thrust::constant_iterator<int> heightIterType;

  // Allows for a 2d window of data with a source value and its 8 surrounding neighbors
  typedef typename thrust::tuple<dataIter, dataIter, dataIter,
                                  dataIter, dataIter, dataIter,
                                  dataIter, dataIter, dataIter> windowTuple;

  // zipped iterator for the above window tuple
  typedef typename thrust::zip_iterator<windowTuple> windowTupleIter;


  // 1, 2, 3 or 4 source windows
  typedef typename thrust::zip_iterator<thrust::tuple<windowTupleIter> > tuple1window;
  typedef typename thrust::zip_iterator<thrust::tuple<windowTupleIter,
                                                      windowTupleIter> > tuple2window;
  typedef typename thrust::zip_iterator<thrust::tuple<windowTupleIter,
                                                      windowTupleIter,
                                                      windowTupleIter> > tuple3window;
  typedef typename thrust::zip_iterator<thrust::tuple<windowTupleIter,
                                                      windowTupleIter,
                                                      windowTupleIter,
                                                      windowTupleIter> > tuple4window;

  // combined tuple for all of the information that we need in our 2d kernels
  typedef typename thrust::tuple<resultIter, tuple1window, indexIterType, widthIterType, heightIterType> Kernel1Window;
  typedef typename thrust::tuple<resultIter, tuple2window, indexIterType, widthIterType, heightIterType> Kernel2Window;
  typedef typename thrust::tuple<resultIter, tuple3window, indexIterType, widthIterType, heightIterType> Kernel3Window;
  typedef typename thrust::tuple<resultIter, tuple4window, indexIterType, widthIterType, heightIterType> Kernel4Window;

  // iterator for this information to be sent in correctly
  typedef typename thrust::zip_iterator<Kernel1Window> Kernel1WindowIter;
  typedef typename thrust::zip_iterator<Kernel2Window> Kernel2WindowIter;
  typedef typename thrust::zip_iterator<Kernel3Window> Kernel3WindowIter;
  typedef typename thrust::zip_iterator<Kernel4Window> Kernel4WindowIter;


  struct randoms_helper_functor {
    int minValue;
    int maxValue;

    randoms_helper_functor(int minValue_, int maxValue_) : minValue(minValue_), maxValue(maxValue_) {}

    __host__ __device__
    void operator()(int &value)
    {
      value = (value % (maxValue - minValue)) + minValue;
    }
  };


  /*********************************************************
   * DeviceData contains two device vectors of the same size
   * and allows for the active one to be swapped at any time.
   * it also manages when these things should be deleted */
  struct DeviceData
  {
    DeviceData(int width_, int height_) : width(width_), height(height_),
    wrap(true) {
      int length = width*height;
      srand(NULL);

      data1 = thrust::device_vector<T>(length);
      data2 = thrust::device_vector<T>(length);

      indexIter = indexIterType(0);
      widthIter = widthIterType(width);
      heightIter = heightIterType(height);

      mainData = &data1;
      alternateData = &data2;

      static unsigned int lastUid = 0;
      uid = lastUid;
      lastUid++;
    }

    int uid;
    int width;
    int height;
    bool wrap;

    indexIterType indexIter;
    widthIterType widthIter;
    heightIterType heightIter;

    thrust::device_vector<T> data1;
    thrust::device_vector<T> data2;

    thrust::device_vector<T> *mainData;
    thrust::device_vector<T> *alternateData;

    bool operator==(const DeviceData &other) const {
      return this->uid == other.uid;
    }

    // A CPU-driven random function that fills the data
    void randomize(int minValue, int maxValue) {
      // Create host vector to hold the random numbers
      thrust::host_vector<T> randoms(width*height);

      // Generate the random numbers
      thrust::generate(randoms.begin(), randoms.end(), rand);

      // Copy data from CPU to GPU
      *mainData = randoms;

      thrust::for_each(mainData->begin(),
                       mainData->end(),
                       randoms_helper_functor(minValue, maxValue));
    }

    void randomize(int maxValue=INT_MAX) {
      randomize(0, maxValue);
    }


    void maybeWrap() {
      if (!wrap) {
        return;
      }

      T *raw_pointer = thrust::raw_pointer_cast(&((*this->mainData)[0]));

      // TODO: make dynamic
      // Maximum cuda block dimension, we should really get this dynamically
      int maxBlockDimensionSize = 512;
      int largestSide = max(this->width, this->height);
      largestSide = max(largestSide, 5); // ensure that we get all 4 corners starting from 1

      int blockWidth;
      int gridWidth;

      if (largestSide > maxBlockDimensionSize) {
        blockWidth = maxBlockDimensionSize;
        gridWidth = 1 + largestSide / maxBlockDimensionSize;
      } else {
        blockWidth = largestSide;
        gridWidth = 1;
      }

      copyWrapAroundAreas<<<dim3(gridWidth, 5), dim3(blockWidth, 1, 1)>>>(raw_pointer, this->width, this->height);
    }

    void swap() {
      if (mainData == &data1) {
        mainData = &data2;
        alternateData = &data1;
      } else {
        mainData = &data1;
        alternateData = &data2;
      }
    }

    thrust::device_vector<T>* unpadded() {
      // Start but ignore the padding
      for (int row = 1; row < height-1; row++) {
      int unpaddedOffset = (row-1)*(width-2);
      int paddedOffset = row*width+1;
      thrust::copy(mainData->begin()+paddedOffset, mainData->begin()+paddedOffset+width-2,
                  alternateData->begin()+unpaddedOffset);
      }

      return alternateData;
    }

    void loadWithUnpaddedData(thrust::device_vector<T> &unpaddedData) {
      // Start but ignore the padding
      for (int row = 0; row < height-2; row++) {
        int unpaddedOffset = row*(width-2);
        int paddedOffset = (row+1)*width+1;
        thrust::copy(unpaddedData.begin()+unpaddedOffset, unpaddedData.begin()+unpaddedOffset+width-2, mainData->begin()+paddedOffset);
      }
    }
  };



  // 2d shifted iterator function to compute the correct one-dimensional offset based on
  // the 2d array that the iterator represents.
  static dataIter shiftedIterator(dataIter iterator, int xChange, int yChange, int width) {
    iterator += yChange*width;
    iterator += xChange;

    return iterator;
  }

  static windowTuple windowStartTuple(DeviceData &data) {
    // Thinking about top left
    // we have a linear array that is has 1 dimension. To convert this to a 2d representation
    // we need to perform some pointer math
    // First observe the following, which is the top-left part of the 2d array
    //  ___________
    // |a b c + + +
    // |d e f + + +
    // |g h i + + +
    // |+ + + + + +
    // |+ + + + + +
    //
    // a is at array[0], b is array[1], c is array[2]
    // e is the point that we are looking at but it is way over at array[1*width+1]
    // g is at array[1*width+0]
    // i, the last point is at array[2*width+2]
    // these values are passed into shiftedIterator

    dataIter currentBegin = data.mainData->begin();
    int width = data.width;

    dataIter topLeft     = shiftedIterator(currentBegin, 0, 0, width);
    dataIter top         = shiftedIterator(currentBegin, 1, 0, width);
    dataIter topRight    = shiftedIterator(currentBegin, 2, 0, width);
    dataIter left        = shiftedIterator(currentBegin, 0, 1, width);
    dataIter center      = shiftedIterator(currentBegin, 1, 1, width);
    dataIter right       = shiftedIterator(currentBegin, 2, 1, width);
    dataIter bottomLeft  = shiftedIterator(currentBegin, 0, 2, width);
    dataIter bottom      = shiftedIterator(currentBegin, 1, 2, width);
    dataIter bottomRight = shiftedIterator(currentBegin, 2, 2, width);

    windowTuple begin = thrust::make_tuple(topLeft,    top,    topRight,
                                                 left,       center, right,
                                                 bottomLeft, bottom, bottomRight);
    return begin;
  }

  // End Iterators
  static windowTuple windowEndTuple(DeviceData &data) {
    dataIter currentEnd = data.mainData->end();
    int width = data.width;

    dataIter topLeft     = shiftedIterator(currentEnd, -2, -2, width);
    dataIter top         = shiftedIterator(currentEnd, -1, -2, width);
    dataIter topRight    = shiftedIterator(currentEnd, -0, -2, width);
    dataIter left        = shiftedIterator(currentEnd, -2, -1, width);
    dataIter center      = shiftedIterator(currentEnd, -1, -1, width);
    dataIter right       = shiftedIterator(currentEnd, -0, -1, width);
    dataIter bottomLeft  = shiftedIterator(currentEnd, -2, -0, width);
    dataIter bottom      = shiftedIterator(currentEnd, -1, -0, width);
    dataIter bottomRight = shiftedIterator(currentEnd, -0, -0, width);

    windowTuple end = thrust::make_tuple(topLeft,    top,    topRight,
                                               left,       center, right,
                                               bottomLeft, bottom, bottomRight);
    return end;
  }

  template <typename K>
  static thrust::zip_iterator<thrust::tuple<resultIter, K, indexIterType, widthIterType, heightIterType> > startIterator(DeviceData &destinationData, K dataWindows) {
    
    dataIter result = shiftedIterator(destinationData.alternateData->begin(), 1, 1, destinationData.width);

    indexIterType index = destinationData.indexIter + destinationData.width+1;
    widthIterType width = destinationData.widthIter;
    heightIterType height = destinationData.heightIter;

    return thrust::make_zip_iterator(thrust::make_tuple(result, dataWindows, index, width, height));
  }

  static Kernel1WindowIter startIterator(DeviceData &destinationData, DeviceData &sourceData1) {
    windowTupleIter window1 = thrust::make_zip_iterator(windowStartTuple(sourceData1));

    tuple1window windowTupleIterator = thrust::make_zip_iterator(thrust::make_tuple(window1));
    return startIterator<tuple1window>(destinationData, windowTupleIterator);
  }
  static Kernel2WindowIter startIterator(DeviceData &destinationData, DeviceData &sourceData1, DeviceData &sourceData2) {
    windowTupleIter window1 = thrust::make_zip_iterator(windowStartTuple(sourceData1));
    windowTupleIter window2 = thrust::make_zip_iterator(windowStartTuple(sourceData2));

    
    tuple2window windowTupleIterator = thrust::make_zip_iterator(thrust::make_tuple(window1, window2));
    return startIterator<tuple2window>(destinationData, windowTupleIterator);
  }
  static Kernel3WindowIter startIterator(DeviceData &destinationData, DeviceData &sourceData1, DeviceData &sourceData2, DeviceData &sourceData3) {
    windowTupleIter window1 = thrust::make_zip_iterator(windowStartTuple(sourceData1));
    windowTupleIter window2 = thrust::make_zip_iterator(windowStartTuple(sourceData2));
    windowTupleIter window3 = thrust::make_zip_iterator(windowStartTuple(sourceData3));

    tuple3window windowTupleIterator = thrust::make_zip_iterator(thrust::make_tuple(window1, window2, window3));
    return startIterator<tuple3window>(destinationData, windowTupleIterator);
  }
  static Kernel4WindowIter startIterator(DeviceData &destinationData, DeviceData &sourceData1, DeviceData &sourceData2, DeviceData &sourceData3, DeviceData &sourceData4) {
    windowTupleIter window1 = thrust::make_zip_iterator(windowStartTuple(sourceData1));
    windowTupleIter window2 = thrust::make_zip_iterator(windowStartTuple(sourceData2));
    windowTupleIter window3 = thrust::make_zip_iterator(windowStartTuple(sourceData3));
    windowTupleIter window4 = thrust::make_zip_iterator(windowStartTuple(sourceData4));

    tuple4window windowTupleIterator = thrust::make_zip_iterator(thrust::make_tuple(window1, window2, window3, window4));
    return startIterator<tuple4window>(destinationData, windowTupleIterator);
  }


  // End Iterator

  template <typename K>
  static thrust::zip_iterator<thrust::tuple<resultIter, K, indexIterType, widthIterType, heightIterType> > endIterator(DeviceData &destinationData, K dataWindows) {
    
    dataIter result = shiftedIterator(destinationData.alternateData->end(), -1, -1, destinationData.width);

    indexIterType index = destinationData.indexIter + destinationData.width*destinationData.height - destinationData.width-1;
    widthIterType width = destinationData.widthIter;
    heightIterType height = destinationData.heightIter;

    return thrust::make_zip_iterator(thrust::make_tuple(result, dataWindows, index, width, height));
  }

  static Kernel1WindowIter endIterator(DeviceData &destinationData, DeviceData &sourceData1) {
    windowTupleIter window1 = thrust::make_zip_iterator(windowEndTuple(sourceData1));

    tuple1window windowTupleIterator = thrust::make_zip_iterator(thrust::make_tuple(window1));
    return endIterator<tuple1window>(destinationData, windowTupleIterator);
  }
  static Kernel2WindowIter endIterator(DeviceData &destinationData, DeviceData &sourceData1, DeviceData &sourceData2) {
    windowTupleIter window1 = thrust::make_zip_iterator(windowEndTuple(sourceData1));
    windowTupleIter window2 = thrust::make_zip_iterator(windowEndTuple(sourceData2));

    tuple2window windowTupleIterator = thrust::make_zip_iterator(thrust::make_tuple(window1, window2));
    return endIterator<tuple2window>(destinationData, windowTupleIterator);
  }
  static Kernel3WindowIter endIterator(DeviceData &destinationData, DeviceData &sourceData1, DeviceData &sourceData2, DeviceData &sourceData3) {
    windowTupleIter window1 = thrust::make_zip_iterator(windowEndTuple(sourceData1));
    windowTupleIter window2 = thrust::make_zip_iterator(windowEndTuple(sourceData2));
    windowTupleIter window3 = thrust::make_zip_iterator(windowEndTuple(sourceData3));

    tuple3window windowTupleIterator = thrust::make_zip_iterator(thrust::make_tuple(window1, window2, window3));
    return endIterator<tuple3window>(destinationData, windowTupleIterator);
  }
  static Kernel4WindowIter endIterator(DeviceData &destinationData, DeviceData &sourceData1, DeviceData &sourceData2, DeviceData &sourceData3, DeviceData &sourceData4) {
    windowTupleIter window1 = thrust::make_zip_iterator(windowEndTuple(sourceData1));
    windowTupleIter window2 = thrust::make_zip_iterator(windowEndTuple(sourceData2));
    windowTupleIter window3 = thrust::make_zip_iterator(windowEndTuple(sourceData3));
    windowTupleIter window4 = thrust::make_zip_iterator(windowEndTuple(sourceData4));

    tuple4window windowTupleIterator = thrust::make_zip_iterator(thrust::make_tuple(window1, window2, window3, window4));
    return endIterator<tuple4window>(destinationData, windowTupleIterator);
  }
};


enum WrapAroundCondition { WrapAroundConditionCorners = 0,
                           WrapAroundConditionLeft = 1,
                           WrapAroundConditionRight = 2,
                           WrapAroundConditionTop = 3,
                           WrapAroundConditionBottom = 4 };

template <typename T>
__global__ void copyWrapAroundAreas(T *array, int width, int height) {
  // 5 conditions
  //   - corners
  //   - left
  //   - right
  //   - top
  //   - bottom
  WrapAroundCondition condition = (WrapAroundCondition)blockIdx.y;

  // To find the thing to operate on, we unroll a 2d structure into 1d
  // so that we can support 512*512 = 262144 length arrays on my (GeForce 320M) hardware
  int position = blockIdx.x*blockDim.x + threadIdx.x;
  int sourceX;
  int sourceY;
  int destinationX;
  int destinationY;

  switch (condition) {
  case WrapAroundConditionCorners:
    // width == 5
    // first value is at 1
    //
    //  0 1 2 3 4
    //  _________
    // |d c . d c|
    // |b A - B a|
    // |. - - - .|
    // |d C - D c|
    // |b a . b a|
    //  ---------

    switch (position) {
    case 1: // top left
      sourceX = 1;
      sourceY = 1;
      destinationX = width - 1;
      destinationY = height - 1;
      break;
    case 2: // top right
      sourceX = width - 2;
      sourceY = 1;
      destinationX = 0;
      destinationY = height - 1;
      break;
    case 3: // bottom left
      sourceX = 1;
      sourceY = height - 2;
      destinationX = width - 1;
      destinationY = 0;
      break;
    case 4: // bottom right
      sourceX = width - 2;
      sourceY = height - 2;
      destinationX = 0;
      destinationY = 0;
      break;
    default:
      return;
    }
    break;
  case WrapAroundConditionLeft: // take from left and insert at right
    sourceX = 1;
    sourceY = position;
    destinationX = width-1;
    destinationY = position;
    break;
  case WrapAroundConditionRight: // take from right and insert at left
    sourceX = width-2;
    sourceY = position;
    destinationX = 0;
    destinationY = position;
    break;
  case WrapAroundConditionTop: // take from top and insert at bottom
    sourceX = position;
    sourceY = 1;
    destinationX = position;
    destinationY = height-1;
    break;
  case WrapAroundConditionBottom: // take from bottom and insert at top
    sourceX = position;
    sourceY = height-2;
    destinationX = position;
    destinationY = 0;
    break;
  }


  if (sourceX >= (width-1) || sourceY >= (height-1) || destinationX > (width-1) || destinationY > (height-1)) {
    return;
  }

  int sourceIndex = sourceY*width + sourceX;
  int destinationIndex = destinationY*width + destinationX;

  array[destinationIndex] = array[sourceIndex];
}

"""


main_template = """\
int main(void)
{
%(main_code)s

  return 0;
}
"""

from parser import parse
from nodes import *


cuda_directory='/usr/local/cuda/'
cuda_compiler=cuda_directory+'bin/nvcc'

cuda_compile_pass1 = """%(cuda_compiler)s -M -D__CUDACC__ %(input_file)s -o %(input_file)s.o.NVCC-depend -m64 -Xcompiler ,\"-g\" -DNVCC -I/usr/local/cuda/include -I/usr/local/cuda/include  -I/usr/include"""

cuda_compile_pass2 = """%(cuda_compiler)s %(input_file)s -c -o %(input_file)s.o -m64 -Xcompiler ,\"-g\" -DNVCC -I/usr/local/cuda/include -I/usr/local/cuda/include -I/usr/include"""

cuda_compile_pass3 = """/usr/bin/c++   -O2 -g -Wl,-search_paths_first -headerpad_max_install_names  ./%(input_file)s.o -o %(output_file)s /usr/local/cuda/lib/libcudart.dylib -Wl,-rpath -Wl,/usr/local/cuda/lib /usr/local/cuda/lib/libcuda.dylib"""


def compile(ast):
  print ast
  functions = []
  main_function_statements = []
  for program_node in ast:
    if type(program_node) in [SequentialFunctionDeclaration,
            ParallelFunctionDeclaration]:
      functions.append(program_node.to_cpp())
    elif type(program_node) == DataDeclaration:
      main_function_statements.append(program_node.to_cpp())
    elif type(program_node) == VariableDeclaration:
      main_function_statements.append(program_node.to_cpp())
    else:
      main_function_statements.append(program_node.to_cpp())

  main_function = main_template % { 'main_code' : indent('\n'.join(main_function_statements)) }

  thrust_code = preamble + '\n'.join(functions) + main_function
  return thrust_code

def main():
  import sys, os
  import shlex, subprocess

  with open(sys.argv[1], 'r') as f:
    code = ''.join(f.readlines())

  ast = parse(code)
  thrust_code = compile(ast)

  with open(sys.argv[2]+'.cu', 'w') as f:
    f.write(thrust_code)


  env = { 'cuda_compiler' : cuda_compiler,
          'input_file' : sys.argv[2]+'.cu',
          'output_file' : sys.argv[2] }

  print('compiling...')
  pass1 = subprocess.Popen(shlex.split(cuda_compile_pass1 % env))
  pass1.wait()
  print('stage one complete')

  pass2 = subprocess.Popen(shlex.split(cuda_compile_pass2 % env))
  pass2.wait()
  print('stage two complete')

  pass3 = subprocess.Popen(shlex.split(cuda_compile_pass3 % env))
  pass3.wait()
  print('stage three complete')

  #os.remove(sys.argv[2]+'.cu')
  os.remove(sys.argv[2]+'.cu.o')
  os.remove(sys.argv[2]+'.cu.o.NVCC-depend')


if __name__ == '__main__':
  main()

