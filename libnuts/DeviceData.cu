#include "DeviceData.h"

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
void printFullArray2D(const thrust::host_vector<T> &vector, int width, int height) {
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
void printArray2D(const thrust::host_vector<T> &vector, int width, int height, int padding) {
  int paddedWidth = width + 2*padding;
  //int paddedHeight = height + 2*padding;

  for (int y=padding; y<height+padding; y++) {
    for (int x=padding; x<width+padding; x++) {
      int i = y*paddedWidth + x;
      std::cout << ((vector[i] == 0) ? "." : stringFromInt(vector[i])) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

  // 2d shifted iterator function to compute the correct one-dimensional offset based on
  // the 2d array that the iterator represents.
template <typename T>
typename ChestnutDetail<T>::dataIter ChestnutDetail<T>::shiftedIterator(typename ChestnutDetail<T>::dataIter iterator, int xChange, int yChange, int width) {
  iterator += yChange*width;
  iterator += xChange;

  return iterator;
}

//  /*********************************************************
//   * ChestnutDetail<T>::DeviceData contains two device vectors of the same size
//   * and allows for the active one to be swapped at any time.
//   * it also manages when these things should be deleted */
//  struct ChestnutDetail<T>::DeviceData
//  {
//    ChestnutDetail<T>::DeviceData(int width_, int height_) : width(width_), height(height_),
//    wrap(true) {
//      int length = width*height;
//      srand(NULL);

//      data1 = thrust::device_vector<T>(length);
//      data2 = thrust::device_vector<T>(length);

//      indexIter = ChestnutDetail<T>::indexIterType(0);
//      widthIter = ChestnutDetail<T>::widthIterType(width);
//      heightIter = ChestnutDetail<T>::heightIterType(height);

//      mainData = &data1;
//      alternateData = &data2;

//      static unsigned int lastUid = 0;
//      uid = lastUid;
//      lastUid++;
//    }

//    int uid;
//    int width;
//    int height;
//    bool wrap;

//    ChestnutDetail<T>::indexIterType indexIter;
//    ChestnutDetail<T>::widthIterType widthIter;
//    ChestnutDetail<T>::heightIterType heightIter;

//    thrust::device_vector<T> data1;
//    thrust::device_vector<T> data2;

//    thrust::device_vector<T> *mainData;
//    thrust::device_vector<T> *alternateData;

//    bool operator==(const ChestnutDetail<T>::DeviceData &other) const {
//      return this->uid == other.uid;
//    }

//    // A CPU-driven random function that fills the data
//    void randomize(int minValue, int maxValue) {
//      // Create host vector to hold the random numbers
//      thrust::host_vector<T> randoms(width*height);

//      // Generate the random numbers
//      thrust::generate(randoms.begin(), randoms.end(), rand);

//      // Copy data from CPU to GPU
//      *mainData = randoms;

//      thrust::for_each(mainData->begin(),
//                       mainData->end(),
//                       randoms_helper_functor(minValue, maxValue));
//    }

//    void randomize(int maxValue=INT_MAX) {
//      randomize(0, maxValue);
//    }


//    void maybeWrap() {
//      if (!wrap) {
//        return;
//      }

//      T *raw_pointer = thrust::raw_pointer_cast(&((*this->mainData)[0]));

//      // TODO: make dynamic
//      // Maximum cuda block dimension, we should really get this dynamically
//      int maxBlockDimensionSize = 512;
//      int largestSide = max(this->width, this->height);
//      largestSide = max(largestSide, 5); // ensure that we get all 4 corners starting from 1

//      int blockWidth;
//      int gridWidth;

//      if (largestSide > maxBlockDimensionSize) {
//        blockWidth = maxBlockDimensionSize;
//        gridWidth = 1 + largestSide / maxBlockDimensionSize;
//      } else {
//        blockWidth = largestSide;
//        gridWidth = 1;
//      }

//      copyWrapAroundAreas<<<dim3(gridWidth, 5), dim3(blockWidth, 1, 1)>>>(raw_pointer, this->width, this->height);
//    }

//    void swap() {
//      if (mainData == &data1) {
//        mainData = &data2;
//        alternateData = &data1;
//      } else {
//        mainData = &data1;
//        alternateData = &data2;
//      }
//    }

//    thrust::device_vector<T>* unpadded() {
//      // Start but ignore the padding
//      for (int row = 1; row < height-1; row++) {
//      int unpaddedOffset = (row-1)*(width-2);
//      int paddedOffset = row*width+1;
//      thrust::copy(mainData->begin()+paddedOffset, mainData->begin()+paddedOffset+width-2,
//                  alternateData->begin()+unpaddedOffset);
//      }

//      return alternateData;
//    }

//    void loadWithUnpaddedData(thrust::device_vector<T> &unpaddedData) {
//      // Start but ignore the padding
//      for (int row = 0; row < height-2; row++) {
//        int unpaddedOffset = row*(width-2);
//        int paddedOffset = (row+1)*width+1;
//        thrust::copy(unpaddedData.begin()+unpaddedOffset, unpaddedData.begin()+unpaddedOffset+width-2, mainData->begin()+paddedOffset);
//      }
//    }
//  };

template <typename T>
typename ChestnutDetail<T>::windowTuple
ChestnutDetail<T>::windowStartTuple(typename ChestnutDetail<T>::DeviceData &data) {
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

    ChestnutDetail<T>::dataIter currentBegin = data.mainData->begin();
    int width = data.width;

    ChestnutDetail<T>::dataIter topLeft     = shiftedIterator(currentBegin, 0, 0, width);
    ChestnutDetail<T>::dataIter top         = shiftedIterator(currentBegin, 1, 0, width);
    ChestnutDetail<T>::dataIter topRight    = shiftedIterator(currentBegin, 2, 0, width);
    ChestnutDetail<T>::dataIter left        = shiftedIterator(currentBegin, 0, 1, width);
    ChestnutDetail<T>::dataIter center      = shiftedIterator(currentBegin, 1, 1, width);
    ChestnutDetail<T>::dataIter right       = shiftedIterator(currentBegin, 2, 1, width);
    ChestnutDetail<T>::dataIter bottomLeft  = shiftedIterator(currentBegin, 0, 2, width);
    ChestnutDetail<T>::dataIter bottom      = shiftedIterator(currentBegin, 1, 2, width);
    ChestnutDetail<T>::dataIter bottomRight = shiftedIterator(currentBegin, 2, 2, width);

    typename ChestnutDetail<T>::windowTuple begin = thrust::make_tuple(topLeft,    top,    topRight,
                                           left,       center, right,
                                           bottomLeft, bottom, bottomRight);
    return begin;
  }

  // End Iterators
template <typename T>
typename ChestnutDetail<T>::windowTuple
ChestnutDetail<T>::windowEndTuple(ChestnutDetail<T>::DeviceData &data) {
    ChestnutDetail<T>::dataIter currentEnd = data.mainData->end();
    int width = data.width;

    ChestnutDetail<T>::dataIter topLeft     = shiftedIterator(currentEnd, -2, -2, width);
    ChestnutDetail<T>::dataIter top         = shiftedIterator(currentEnd, -1, -2, width);
    ChestnutDetail<T>::dataIter topRight    = shiftedIterator(currentEnd, -0, -2, width);
    ChestnutDetail<T>::dataIter left        = shiftedIterator(currentEnd, -2, -1, width);
    ChestnutDetail<T>::dataIter center      = shiftedIterator(currentEnd, -1, -1, width);
    ChestnutDetail<T>::dataIter right       = shiftedIterator(currentEnd, -0, -1, width);
    ChestnutDetail<T>::dataIter bottomLeft  = shiftedIterator(currentEnd, -2, -0, width);
    ChestnutDetail<T>::dataIter bottom      = shiftedIterator(currentEnd, -1, -0, width);
    ChestnutDetail<T>::dataIter bottomRight = shiftedIterator(currentEnd, -0, -0, width);

    typename ChestnutDetail<T>::windowTuple end = thrust::make_tuple(topLeft,    top,    topRight,
                                         left,       center, right,
                                         bottomLeft, bottom, bottomRight);
    return end;
  }

// Start iter template
#define return_start_iter(windowTupleIterator) \
ChestnutDetail<T>::dataIter result = shiftedIterator(destinationData.alternateData->begin(), 1, 1, destinationData.width); \
\
ChestnutDetail<T>::indexIterType index   = destinationData.indexIter + destinationData.width+1; \
ChestnutDetail<T>::widthIterType width   = destinationData.widthIter; \
ChestnutDetail<T>::heightIterType height = destinationData.heightIter; \
\
return thrust::make_zip_iterator(thrust::make_tuple(result, windowTupleIterator, index, width, height));

// End iter template
#define return_end_iter(windowTupleIterator) \
ChestnutDetail<T>::dataIter result = shiftedIterator(destinationData.alternateData->end(), -1, -1, destinationData.width); \
\
ChestnutDetail<T>::indexIterType index = destinationData.indexIter + destinationData.width*destinationData.height - destinationData.width-1; \
ChestnutDetail<T>::widthIterType width = destinationData.widthIter; \
ChestnutDetail<T>::heightIterType height = destinationData.heightIter; \
\
return thrust::make_zip_iterator(thrust::make_tuple(result, windowTupleIterator, index, width, height)); \

  template <typename T> typename ChestnutDetail<T>::Kernel0WindowIter ChestnutDetail<T>::startIterator(ChestnutDetail<T>::DeviceData &destinationData) {
    ChestnutDetail<T>::tuple0Window windowTupleIterator(0);

    return_start_iter(windowTupleIterator);
  }

  template <typename T> typename ChestnutDetail<T>::Kernel1WindowIter ChestnutDetail<T>::startIterator(ChestnutDetail<T>::DeviceData &destinationData, ChestnutDetail<T>::DeviceData &sourceData1) {
    ChestnutDetail<T>::windowTupleIter window1 = thrust::make_zip_iterator(windowStartTuple(sourceData1));

    ChestnutDetail<T>::tuple1Window windowTupleIterator = thrust::make_zip_iterator(thrust::make_tuple(window1));
    return_start_iter(windowTupleIterator);
}
  template <typename T> typename ChestnutDetail<T>::Kernel2WindowIter ChestnutDetail<T>::startIterator(ChestnutDetail<T>::DeviceData &destinationData, ChestnutDetail<T>::DeviceData &sourceData1, ChestnutDetail<T>::DeviceData &sourceData2) {
    ChestnutDetail<T>::windowTupleIter window1 = thrust::make_zip_iterator(windowStartTuple(sourceData1));
    ChestnutDetail<T>::windowTupleIter window2 = thrust::make_zip_iterator(windowStartTuple(sourceData2));


    ChestnutDetail<T>::tuple2Window windowTupleIterator = thrust::make_zip_iterator(thrust::make_tuple(window1, window2));
    return_start_iter(windowTupleIterator);
}
  template <typename T> typename ChestnutDetail<T>::Kernel3WindowIter ChestnutDetail<T>::startIterator(ChestnutDetail<T>::DeviceData &destinationData, ChestnutDetail<T>::DeviceData &sourceData1, ChestnutDetail<T>::DeviceData &sourceData2, ChestnutDetail<T>::DeviceData &sourceData3) {
    ChestnutDetail<T>::windowTupleIter window1 = thrust::make_zip_iterator(windowStartTuple(sourceData1));
    ChestnutDetail<T>::windowTupleIter window2 = thrust::make_zip_iterator(windowStartTuple(sourceData2));
    ChestnutDetail<T>::windowTupleIter window3 = thrust::make_zip_iterator(windowStartTuple(sourceData3));

    ChestnutDetail<T>::tuple3Window windowTupleIterator = thrust::make_zip_iterator(thrust::make_tuple(window1, window2, window3));
    return_start_iter(windowTupleIterator);
}
  template <typename T> typename ChestnutDetail<T>::Kernel4WindowIter ChestnutDetail<T>::startIterator(ChestnutDetail<T>::DeviceData &destinationData, ChestnutDetail<T>::DeviceData &sourceData1, ChestnutDetail<T>::DeviceData &sourceData2, ChestnutDetail<T>::DeviceData &sourceData3, ChestnutDetail<T>::DeviceData &sourceData4) {
    ChestnutDetail<T>::windowTupleIter window1 = thrust::make_zip_iterator(windowStartTuple(sourceData1));
    ChestnutDetail<T>::windowTupleIter window2 = thrust::make_zip_iterator(windowStartTuple(sourceData2));
    ChestnutDetail<T>::windowTupleIter window3 = thrust::make_zip_iterator(windowStartTuple(sourceData3));
    ChestnutDetail<T>::windowTupleIter window4 = thrust::make_zip_iterator(windowStartTuple(sourceData4));

    ChestnutDetail<T>::tuple4Window windowTupleIterator = thrust::make_zip_iterator(thrust::make_tuple(window1, window2, window3, window4));
    return_start_iter(windowTupleIterator);
}


  // End Iterator

  template <typename T> typename ChestnutDetail<T>::Kernel0WindowIter ChestnutDetail<T>::endIterator(ChestnutDetail<T>::DeviceData &destinationData) {
    ChestnutDetail<T>::tuple0Window windowTupleIterator(0);
    return_end_iter(windowTupleIterator);
  }

  template <typename T> typename ChestnutDetail<T>::Kernel1WindowIter ChestnutDetail<T>::endIterator(ChestnutDetail<T>::DeviceData &destinationData, ChestnutDetail<T>::DeviceData &sourceData1) {
    ChestnutDetail<T>::windowTupleIter window1 = thrust::make_zip_iterator(windowEndTuple(sourceData1));

    ChestnutDetail<T>::tuple1Window windowTupleIterator = thrust::make_zip_iterator(thrust::make_tuple(window1));
    return_end_iter(windowTupleIterator);
}
  template <typename T> typename ChestnutDetail<T>::Kernel2WindowIter ChestnutDetail<T>::endIterator(ChestnutDetail<T>::DeviceData &destinationData, ChestnutDetail<T>::DeviceData &sourceData1, ChestnutDetail<T>::DeviceData &sourceData2) {
    ChestnutDetail<T>::windowTupleIter window1 = thrust::make_zip_iterator(windowEndTuple(sourceData1));
    ChestnutDetail<T>::windowTupleIter window2 = thrust::make_zip_iterator(windowEndTuple(sourceData2));

    ChestnutDetail<T>::tuple2Window windowTupleIterator = thrust::make_zip_iterator(thrust::make_tuple(window1, window2));
    return_end_iter(windowTupleIterator);
  }
  template <typename T> typename ChestnutDetail<T>::Kernel3WindowIter ChestnutDetail<T>::endIterator(ChestnutDetail<T>::DeviceData &destinationData, ChestnutDetail<T>::DeviceData &sourceData1, ChestnutDetail<T>::DeviceData &sourceData2, ChestnutDetail<T>::DeviceData &sourceData3) {
    ChestnutDetail<T>::windowTupleIter window1 = thrust::make_zip_iterator(windowEndTuple(sourceData1));
    ChestnutDetail<T>::windowTupleIter window2 = thrust::make_zip_iterator(windowEndTuple(sourceData2));
    ChestnutDetail<T>::windowTupleIter window3 = thrust::make_zip_iterator(windowEndTuple(sourceData3));

    ChestnutDetail<T>::tuple3Window windowTupleIterator = thrust::make_zip_iterator(thrust::make_tuple(window1, window2, window3));
    return_end_iter(windowTupleIterator);
  }
  template <typename T> typename ChestnutDetail<T>::Kernel4WindowIter ChestnutDetail<T>::endIterator(ChestnutDetail<T>::DeviceData &destinationData, ChestnutDetail<T>::DeviceData &sourceData1, ChestnutDetail<T>::DeviceData &sourceData2, ChestnutDetail<T>::DeviceData &sourceData3, ChestnutDetail<T>::DeviceData &sourceData4) {
    ChestnutDetail<T>::windowTupleIter window1 = thrust::make_zip_iterator(windowEndTuple(sourceData1));
    ChestnutDetail<T>::windowTupleIter window2 = thrust::make_zip_iterator(windowEndTuple(sourceData2));
    ChestnutDetail<T>::windowTupleIter window3 = thrust::make_zip_iterator(windowEndTuple(sourceData3));
    ChestnutDetail<T>::windowTupleIter window4 = thrust::make_zip_iterator(windowEndTuple(sourceData4));

    ChestnutDetail<T>::tuple4Window windowTupleIterator = thrust::make_zip_iterator(thrust::make_tuple(window1, window2, window3, window4));
    return_end_iter(windowTupleIterator);
  }

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

#define iStruct(name, datatype) template struct name<datatype>
#define initAllStructTypes(name) iStruct(name, int); iStruct(name, char); iStruct(name, float); iStruct(name, uchar4)

//initAllStructTypes(vector);
//initAllStructTypes(dataIter);
//initAllStructTypes(ChestnutDetail<T>::resultIter);
//initAllStructTypes(windowTuple);
//initAllStructTypes(windowTupleIter);

//initAllStructTypes(tuple1window);
//initAllStructTypes(tuple2window);
//initAllStructTypes(tuple3window);
//initAllStructTypes(tuple4window);

//initAllStructTypes(ChestnutDetail<T>::Kernel0Window);
//initAllStructTypes(ChestnutDetail<T>::Kernel1Window);
//initAllStructTypes(ChestnutDetail<T>::Kernel2Window);
//initAllStructTypes(ChestnutDetail<T>::Kernel3Window);
//initAllStructTypes(ChestnutDetail<T>::Kernel4Window);

//initAllStructTypes(ChestnutDetail<T>::Kernel0WindowIter);
//initAllStructTypes(ChestnutDetail<T>::Kernel1WindowIter);
//initAllStructTypes(ChestnutDetail<T>::Kernel2WindowIter);
//initAllStructTypes(ChestnutDetail<T>::Kernel3WindowIter);
//initAllStructTypes(ChestnutDetail<T>::Kernel4WindowIter);

//initAllStructTypes(DeviceData);

#define initFunctionsWithType(T) \
\
template void copyWrapAroundAreas<T>(T*, int, int); \
\
template ChestnutDetail<T>::dataIter ChestnutDetail<T>::shiftedIterator(ChestnutDetail<T>::dataIter iterator, int xChange, int yChange, int width); \
\
template ChestnutDetail<T>::windowTuple ChestnutDetail<T>::windowStartTuple(ChestnutDetail<T>::DeviceData &data); \
\
template ChestnutDetail<T>::Kernel0WindowIter ChestnutDetail<T>::startIterator(ChestnutDetail<T>::DeviceData &destinationData); \
template ChestnutDetail<T>::Kernel1WindowIter ChestnutDetail<T>::startIterator(ChestnutDetail<T>::DeviceData &destinationData, ChestnutDetail<T>::DeviceData &sourceData1); \
template ChestnutDetail<T>::Kernel2WindowIter ChestnutDetail<T>::startIterator(ChestnutDetail<T>::DeviceData &destinationData, ChestnutDetail<T>::DeviceData &sourceData1, ChestnutDetail<T>::DeviceData &sourceData2); \
template ChestnutDetail<T>::Kernel3WindowIter ChestnutDetail<T>::startIterator(ChestnutDetail<T>::DeviceData &destinationData, ChestnutDetail<T>::DeviceData &sourceData1, ChestnutDetail<T>::DeviceData &sourceData2, ChestnutDetail<T>::DeviceData &sourceData3); \
template ChestnutDetail<T>::Kernel4WindowIter ChestnutDetail<T>::startIterator(ChestnutDetail<T>::DeviceData &destinationData, ChestnutDetail<T>::DeviceData &sourceData1, ChestnutDetail<T>::DeviceData &sourceData2, ChestnutDetail<T>::DeviceData &sourceData3, ChestnutDetail<T>::DeviceData &sourceData4); \
\
template ChestnutDetail<T>::windowTuple ChestnutDetail<T>::windowEndTuple(ChestnutDetail<T>::DeviceData &data); \
\
template ChestnutDetail<T>::Kernel0WindowIter ChestnutDetail<T>::endIterator(ChestnutDetail<T>::DeviceData &destinationData); \
template ChestnutDetail<T>::Kernel1WindowIter ChestnutDetail<T>::endIterator(ChestnutDetail<T>::DeviceData &destinationData, ChestnutDetail<T>::DeviceData &sourceData1); \
template ChestnutDetail<T>::Kernel2WindowIter ChestnutDetail<T>::endIterator(ChestnutDetail<T>::DeviceData &destinationData, ChestnutDetail<T>::DeviceData &sourceData1, ChestnutDetail<T>::DeviceData &sourceData2); \
template ChestnutDetail<T>::Kernel3WindowIter ChestnutDetail<T>::endIterator(ChestnutDetail<T>::DeviceData &destinationData, ChestnutDetail<T>::DeviceData &sourceData1, ChestnutDetail<T>::DeviceData &sourceData2, ChestnutDetail<T>::DeviceData &sourceData3); \
template ChestnutDetail<T>::Kernel4WindowIter ChestnutDetail<T>::endIterator(ChestnutDetail<T>::DeviceData &destinationData, ChestnutDetail<T>::DeviceData &sourceData1, ChestnutDetail<T>::DeviceData &sourceData2, ChestnutDetail<T>::DeviceData &sourceData3, ChestnutDetail<T>::DeviceData &sourceData4);

//template ChestnutDetail<int>::Kernel1WindowIter ChestnutDetail<int>::endIterator(ChestnutDetail<int>::DeviceData&, ChestnutDetail<int>::DeviceData&);

#define initPrintFunctionsWithType(T) \
template void printFullArray2D(const thrust::host_vector<T> &vector, int width, int height); \
template void printArray2D(const thrust::host_vector<T> &vector, int width, int height, int padding); \

initFunctionsWithType(int);
initFunctionsWithType(char);
initFunctionsWithType(float);
initFunctionsWithType(uchar4)

initPrintFunctionsWithType(int);
initPrintFunctionsWithType(char);
initPrintFunctionsWithType(float);
