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

#ifndef __device__
#define __device__
#endif

#ifndef __global__
#define __global__
#endif

typedef typename thrust::device_vector<float>::iterator                     FloatIterator;
typedef typename thrust::tuple<FloatIterator, FloatIterator, FloatIterator> FloatIteratorTuple;
typedef typename thrust::zip_iterator<FloatIteratorTuple>                   Float3Iterator;

typedef typename thrust::device_vector<int>                                 IntVector;
typedef typename thrust::device_vector<int>::iterator                       IntIterator;
typedef typename thrust::tuple<IntIterator, IntIterator, IntIterator,
                               IntIterator, IntIterator, IntIterator,
                               IntIterator, IntIterator, IntIterator, IntIterator>       IntIterator9Tuple;
typedef typename thrust::zip_iterator<IntIterator9Tuple>                    Int9Iterator;

std::string stringFromInt(int number)
{
   std::stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}



struct game_of_life_functor
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        // result[i] = left[i] + right[i] + top[i] + bottom[i] + tl + tr + bl + br;


        int neighborCount = thrust::get<1>(t) + thrust::get<2>(t) + thrust::get<3>(t)
                           +thrust::get<4>(t) +                     thrust::get<6>(t)
                           +thrust::get<7>(t) + thrust::get<8>(t) + thrust::get<9>(t);

        int state;

        // if cell is live
        if (thrust::get<5>(t)){
          // if <= 1 neighbor, cell dies from loneliness
          if (neighborCount <= 1){
            state = 0;
          }
          // if >= 4 neighbors, cell dies from overpopulation
          else if (neighborCount >= 4){
            state = 0;
          } else {
            state = 1;
          }
        // if cell is dead
        } else {
          if (neighborCount == 3){
            state = 1;
          } else {
            state = 0;
          }
        }

        thrust::get<0>(t) = state;
    }
};

enum WrapAroundCondition { WrapAroundConditionCorners = 0,
                           WrapAroundConditionLeft = 1,
                           WrapAroundConditionRight = 2,
                           WrapAroundConditionTop = 3,
                           WrapAroundConditionBottom = 4 };

__global__ void copyWrapAroundAreas(int *array, int width, int height) {
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

IntIterator shiftedIterator(IntIterator iterator, int xChange, int yChange, int width) {
  iterator += yChange*width;
  iterator += xChange;

  return iterator;
}

IntIterator9Tuple createStartTuple(IntVector &currentState, IntVector &nextState, int width) {
  IntIterator currentBegin = currentState.begin();
  IntIterator nextBegin = nextState.begin();

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

  IntIterator topLeft     = shiftedIterator(currentBegin, 0, 0, width);
  IntIterator top         = shiftedIterator(currentBegin, 1, 0, width);
  IntIterator topRight    = shiftedIterator(currentBegin, 2, 0, width);
  IntIterator left        = shiftedIterator(currentBegin, 0, 1, width);
  IntIterator center      = shiftedIterator(currentBegin, 1, 1, width);
  IntIterator right       = shiftedIterator(currentBegin, 2, 1, width);
  IntIterator bottomLeft  = shiftedIterator(currentBegin, 0, 2, width);
  IntIterator bottom      = shiftedIterator(currentBegin, 1, 2, width);
  IntIterator bottomRight = shiftedIterator(currentBegin, 2, 2, width);

  IntIterator centerResult = shiftedIterator(nextBegin, 1, 1, width);


  IntIterator9Tuple begin = thrust::make_tuple(centerResult, topLeft, top, topRight,
                                                       left,center ,    right,
                                                       bottomLeft, bottom, bottomRight);

  return begin;
}

IntIterator9Tuple createEndTuple(IntVector &currentState, IntVector &nextState, int width) {
  IntIterator currentEnd = currentState.end();
  IntIterator nextEnd = nextState.end();

  IntIterator topLeft     = shiftedIterator(currentEnd, -2, -2, width);
  IntIterator top         = shiftedIterator(currentEnd, -1, -2, width);
  IntIterator topRight    = shiftedIterator(currentEnd, -0, -2, width);
  IntIterator left        = shiftedIterator(currentEnd, -2, -1, width);
  IntIterator center      = shiftedIterator(currentEnd, -1, -1, width);
  IntIterator right       = shiftedIterator(currentEnd, -0, -1, width);
  IntIterator bottomLeft  = shiftedIterator(currentEnd, -2, -0, width);
  IntIterator bottom      = shiftedIterator(currentEnd, -1, -0, width);
  IntIterator bottomRight = shiftedIterator(currentEnd, -0, -0, width);

  IntIterator centerEndResult  = shiftedIterator(nextEnd, -1, -1, width);

  IntIterator9Tuple end = thrust::make_tuple(centerEndResult, topLeft, top, topRight,
                                             left,center ,    right,
                                             bottomLeft, bottom, bottomRight);

  return end;
}

void printArray2D(const thrust::host_vector<int> &vector, int width, int height) {
  for (int y=0; y<height; y++) {
    for (int x=0; x<width; x++) {
      int i = y*width + x;
      std::cout << ((vector[i] == 0) ? "." : stringFromInt(vector[i])) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main(void)
{
  int width = 30;
  int height = 10;
  int iterations = 100000;
  int percentLivingAtStart = 30;
  bool wrapAround = true;

  int paddedWidth = width+2;
  int paddedHeight = height+2;

  thrust::host_vector<int> hostData(paddedWidth*paddedHeight);
  //thrust::generate(h_vec.begin(), h_vec.end(), rand);

  for (int y=1; y<height+1; y++) {
    for (int x=1; x<width+1; x++) {
      int i = y*paddedWidth + x;

      int randVal = rand() % 100;
      hostData[i] = (randVal < percentLivingAtStart) ? 1 : 0;
    }
  }

  printArray2D(hostData, paddedWidth, paddedHeight);
  std::cout << std::endl;

  // transfer data to the device
  thrust::device_vector<int> currentState = hostData;
  thrust::device_vector<int> nextState(paddedWidth*paddedHeight);

  thrust::zip_iterator<IntIterator9Tuple> currentStateStartIterator
      = thrust::make_zip_iterator(createStartTuple(currentState, nextState, paddedWidth));
  thrust::zip_iterator<IntIterator9Tuple> currentStateEndIterator
      = thrust::make_zip_iterator(createEndTuple(currentState, nextState, paddedWidth));

  thrust::zip_iterator<IntIterator9Tuple> nextStateStartIterator
      = thrust::make_zip_iterator(createStartTuple(nextState, currentState, paddedWidth));
  thrust::zip_iterator<IntIterator9Tuple> nextStateEndIterator
      = thrust::make_zip_iterator(createEndTuple(nextState, currentState, paddedWidth));

  thrust::device_vector<int> temp;
  thrust::zip_iterator<IntIterator9Tuple> tempStartIterator;
  thrust::zip_iterator<IntIterator9Tuple> tempEndIterator;

  for (int iteration=0; iteration<iterations; iteration++) {
    if ((iterations/10 > 0) && iteration % (iterations/10) == 0) {
      std::cout << "Iteration " << iteration << std::endl;
    }

    int *raw_pointer = thrust::raw_pointer_cast(&currentState[0]);

    if (wrapAround) {
      int maxBlockDimensionSize = 512;
      int largestSide = max(paddedWidth, paddedHeight);
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

      copyWrapAroundAreas<<<1, dim3(blockWidth, blockHeight, 5)>>>(raw_pointer, paddedWidth, paddedHeight);
    }

    thrust::for_each(currentStateStartIterator, currentStateEndIterator, game_of_life_functor());

    tempStartIterator = currentStateStartIterator;
    tempEndIterator = currentStateEndIterator;

    currentStateStartIterator = nextStateStartIterator;
    currentStateEndIterator = nextStateEndIterator;
    nextStateStartIterator = tempStartIterator;
    nextStateEndIterator = tempEndIterator;
  }

  // transfer data back to host
  hostData = currentState;


  printArray2D(hostData, paddedWidth, paddedHeight);

  return 0;
}
