/*
 *   Copyright 2009 Andrew Stromme <astromme@chatonka.com>
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU Library General Public License as
 *   published by the Free Software Foundation; either version 2, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details
 *
 *   You should have received a copy of the GNU Library General Public
 *   License along with this program; if not, write to the
 *   Free Software Foundation, Inc.,
 *   51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#ifndef DEVICEDATA_H
#define DEVICEDATA_H

#include "walnut_global.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include <math.h>
#include <limits.h>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <sstream>

WALNUT_EXPORT std::string stringFromInt(int number);

/*
  Simple debugging function to print out a 2d array. Does not know about
  padding, so send it the padded width/height rather than the base width/height
*/
WALNUT_EXPORT template <typename T>
void printFullArray2D(const thrust::host_vector<T> &vector, int width, int height);

/*
   Slightly smarter padding-aware print function. Send it in the width and the
   padding and it will do the dirty work for you.
*/
WALNUT_EXPORT template <typename T>
void printArray2D(const thrust::host_vector<T> &vector, int width, int height, int padding);


WALNUT_EXPORT
enum WrapAroundCondition { WrapAroundConditionCorners = 0,
                           WrapAroundConditionLeft = 1,
                           WrapAroundConditionRight = 2,
                           WrapAroundConditionTop = 3,
                           WrapAroundConditionBottom = 4 };


WALNUT_EXPORT template <typename T>
__global__ void copyWrapAroundAreas(T *array, int width, int height);

struct WALNUT_EXPORT randoms_helper_functor {
  int minValue;
  int maxValue;

  randoms_helper_functor(int minValue_, int maxValue_) : minValue(minValue_), maxValue(maxValue_) {}

  __host__ __device__
  void operator()(int &value)
  {
    value = (value % (maxValue - minValue)) + minValue;
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
struct WALNUT_EXPORT ChestnutDetail {
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


  // 0, 1, 2, 3 or 4 source windows
  typedef typename thrust::constant_iterator<int>                        tuple0Window;
  typedef typename thrust::zip_iterator<thrust::tuple<windowTupleIter> > tuple1Window;
  typedef typename thrust::zip_iterator<thrust::tuple<windowTupleIter,
                                                      windowTupleIter> > tuple2Window;
  typedef typename thrust::zip_iterator<thrust::tuple<windowTupleIter,
                                                      windowTupleIter,
                                                      windowTupleIter> > tuple3Window;
  typedef typename thrust::zip_iterator<thrust::tuple<windowTupleIter,
                                                      windowTupleIter,
                                                      windowTupleIter,
                                                      windowTupleIter> > tuple4Window;

  // combined tuple for all of the information that we need in our 2d kernels
  typedef typename thrust::tuple<resultIter, tuple0Window, indexIterType, widthIterType, heightIterType> Kernel0Window;
  typedef typename thrust::tuple<resultIter, tuple1Window, indexIterType, widthIterType, heightIterType> Kernel1Window;
  typedef typename thrust::tuple<resultIter, tuple2Window, indexIterType, widthIterType, heightIterType> Kernel2Window;
  typedef typename thrust::tuple<resultIter, tuple3Window, indexIterType, widthIterType, heightIterType> Kernel3Window;
  typedef typename thrust::tuple<resultIter, tuple4Window, indexIterType, widthIterType, heightIterType> Kernel4Window;

  // iterator for this information to be sent in correctly
  typedef typename thrust::zip_iterator<Kernel0Window> Kernel0WindowIter;
  typedef typename thrust::zip_iterator<Kernel1Window> Kernel1WindowIter;
  typedef typename thrust::zip_iterator<Kernel2Window> Kernel2WindowIter;
  typedef typename thrust::zip_iterator<Kernel3Window> Kernel3WindowIter;
  typedef typename thrust::zip_iterator<Kernel4Window> Kernel4WindowIter;


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

      data1 = vector(length);
      data2 = vector(length);

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

    vector data1;
    vector data2;

    vector *mainData;
    vector *alternateData;

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
      int largestSide = std::max(this->width, this->height);
      largestSide = std::max(largestSide, 5); // ensure that we get all 4 corners starting from 1

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
  static dataIter shiftedIterator(dataIter iterator, int xChange, int yChange, int width);

  // Start Iterators
  static windowTuple windowStartTuple(DeviceData &data);

  static Kernel0WindowIter startIterator(DeviceData &destinationData);
  static Kernel1WindowIter startIterator(DeviceData &destinationData, DeviceData &sourceData1);
  static Kernel2WindowIter startIterator(DeviceData &destinationData, DeviceData &sourceData1, DeviceData &sourceData2);
  static Kernel3WindowIter startIterator(DeviceData &destinationData, DeviceData &sourceData1, DeviceData &sourceData2, DeviceData &sourceData3);
  static Kernel4WindowIter startIterator(DeviceData &destinationData, DeviceData &sourceData1, DeviceData &sourceData2, DeviceData &sourceData3, DeviceData &sourceData4);

  // End Iterators
  static windowTuple windowEndTuple(DeviceData &data);

  static Kernel0WindowIter endIterator(DeviceData &destinationData);
  static Kernel1WindowIter endIterator(DeviceData &destinationData, DeviceData &sourceData1);
  static Kernel2WindowIter endIterator(DeviceData &destinationData, DeviceData &sourceData1, DeviceData &sourceData2);
  static Kernel3WindowIter endIterator(DeviceData &destinationData, DeviceData &sourceData1, DeviceData &sourceData2, DeviceData &sourceData3);
  static Kernel4WindowIter endIterator(DeviceData &destinationData, DeviceData &sourceData1, DeviceData &sourceData2, DeviceData &sourceData3, DeviceData &sourceData4);
};


#endif // DEVICEDATA_H
