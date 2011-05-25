#!/usr/bin/env python

preamble = """

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
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
      std::cout << ((vector[i] == 0) ? "." : stringFromInt(vector[i])) << " ";
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

/*********************************************************
 * DeviceData contains two device vectors of the same size
 * and allows for the active one to be swapped at any time.
 * it also manages when these things should be deleted */
template <class T>
struct DeviceData
{
  DeviceData(int width_, int height_) : width(width_), height(height_), wrap(false) {
    int length = width*height;

    data1 = thrust::device_vector<T>(length);
    data2 = thrust::device_vector<T>(length);

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

  thrust::device_vector<T> data1;
  thrust::device_vector<T> data2;

  thrust::device_vector<T> *mainData;
  thrust::device_vector<T> *alternateData;

  bool operator==(const DeviceData &other) const {
    return this->uid == other.uid;
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
    int blockHeight;

    if (largestSide > maxBlockDimensionSize) {
      blockWidth = maxBlockDimensionSize;
      blockHeight = 1 + largestSide / maxBlockDimensionSize;
    } else {
      blockWidth = largestSide;
      blockHeight = 1;
    }

    copyWrapAroundAreas<<<1, dim3(blockWidth, blockHeight, 5)>>>(raw_pointer, this->width, this->height);
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
};


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
  typedef typename thrust::device_vector<T> vectorType;
  typedef typename vectorType::iterator iterType;
  // Allows for a destination plus a source value and its 8 surrounding neighbors
  typedef typename thrust::tuple<iterType, iterType, iterType,
                                 iterType, iterType, iterType,
                                 iterType, iterType, iterType,
                                 iterType> tupleType;

  typedef typename thrust::zip_iterator<tupleType> tupleIterType;

  static iterType shiftedIterator(iterType iterator, int xChange, int yChange, int width) {
    iterator += yChange*width;
    iterator += xChange;

    return iterator;
  }

  static tupleType createStartTuple(iterType &currentBegin, iterType &nextBegin, int width) {
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

    iterType topLeft     = shiftedIterator(currentBegin, 0, 0, width);
    iterType top         = shiftedIterator(currentBegin, 1, 0, width);
    iterType topRight    = shiftedIterator(currentBegin, 2, 0, width);
    iterType left        = shiftedIterator(currentBegin, 0, 1, width);
    iterType center      = shiftedIterator(currentBegin, 1, 1, width);
    iterType right       = shiftedIterator(currentBegin, 2, 1, width);
    iterType bottomLeft  = shiftedIterator(currentBegin, 0, 2, width);
    iterType bottom      = shiftedIterator(currentBegin, 1, 2, width);
    iterType bottomRight = shiftedIterator(currentBegin, 2, 2, width);

    iterType centerResult = shiftedIterator(nextBegin, 1, 1, width);


    tupleType begin = thrust::make_tuple(centerResult, topLeft, top, topRight,
                                                        left,center ,    right,
                                                        bottomLeft, bottom, bottomRight);
    return begin;
  }

  static tupleType createStartTuple(DeviceData<T> &sourceData, DeviceData<T> &destinationData) {
    if (sourceData == destinationData) {
      iterType currentBegin = sourceData.mainData->begin();
      iterType nextBegin = sourceData.alternateData->begin();

      return createStartTuple(currentBegin, nextBegin, sourceData.width);
    } else {
      iterType currentBegin = sourceData.mainData->begin();
      iterType nextBegin = destinationData.mainData->begin();

      return createStartTuple(currentBegin, nextBegin, sourceData.width);
    }
  }


  static tupleIterType createStartIterator(DeviceData<T> &sourceData, DeviceData<T> &destinationData) {
    return thrust::make_zip_iterator(createStartTuple(sourceData, destinationData));
  }

  // End Iterators
  static tupleType createEndTuple(iterType &currentEnd, iterType &nextEnd, int width) {
    iterType topLeft     = shiftedIterator(currentEnd, -2, -2, width);
    iterType top         = shiftedIterator(currentEnd, -1, -2, width);
    iterType topRight    = shiftedIterator(currentEnd, -0, -2, width);
    iterType left        = shiftedIterator(currentEnd, -2, -1, width);
    iterType center      = shiftedIterator(currentEnd, -1, -1, width);
    iterType right       = shiftedIterator(currentEnd, -0, -1, width);
    iterType bottomLeft  = shiftedIterator(currentEnd, -2, -0, width);
    iterType bottom      = shiftedIterator(currentEnd, -1, -0, width);
    iterType bottomRight = shiftedIterator(currentEnd, -0, -0, width);

    iterType centerEndResult  = shiftedIterator(nextEnd, -1, -1, width);

    tupleType end = thrust::make_tuple(centerEndResult, topLeft, top, topRight,
                                              left,center ,    right,
                                              bottomLeft, bottom, bottomRight);
    return end;
  }


  static tupleType createEndTuple(DeviceData<T> &sourceData, DeviceData<T> &destinationData) {
    if (sourceData == destinationData) {
      iterType currentEnd = sourceData.mainData->end();
      iterType nextEnd = sourceData.alternateData->end();

      return createEndTuple(currentEnd, nextEnd, sourceData.width);
    } else {
      iterType currentEnd = sourceData.mainData->end();
      iterType nextEnd = destinationData.mainData->end();

      return createEndTuple(currentEnd, nextEnd, sourceData.width);
    }
  }

  static tupleIterType createEndIterator(DeviceData<T> &sourceData, DeviceData<T> &destinationData) {
    return thrust::make_zip_iterator(createEndTuple(sourceData, destinationData));
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
  WrapAroundCondition condition = (WrapAroundCondition)threadIdx.z;

  // To find the thing to operate on, we unroll a 2d structure into 1d
  // so that we can support 512*512 = 262144 length arrays on my (GeForce 320M) hardware
  int position = threadIdx.y*blockDim.x + threadIdx.x;
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


main_template = """
int main(void)
{
  %(main_code)s

  return 0;
}
"""

from parser import parse
from nodes import *

cuda_compiler='/usr/local/cuda/bin/nvcc'

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

  main_function = main_template % { 'main_code' : '\n'.join(main_function_statements) }

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

