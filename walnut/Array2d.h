/*
 *   Copyright 2011 Andrew Stromme <astromme@chatonka.com>
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as
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

#ifndef ARRAY2D_H
#define ARRAY2D_H

#include "walnut_global.h"
#include "Sizes.h"
#include "Color.h"

#include <thrust/sort.h>
#include <thrust/device_vector.h>


__device__ inline bool operator<(const Walnut::Color &left, const Walnut::Color &right) {
  return left.red() + left.green() + left.blue() + left.opacity() < right.red() + right.green() + right.blue() + right.opacity();
}

namespace Walnut {

#define _index thrust::get<0>(_t)
#define _x (thrust::get<0>(_t) % thrust::get<1>(_t))
#define _y (thrust::get<0>(_t) / thrust::get<1>(_t))
#define _width thrust::get<1>(_t)
#define _height thrust::get<2>(_t)

template <typename T>
struct WALNUT_EXPORT Array2d
{
  T *data;
  int width;
  int height;

  typedef T Type;

  __host__ __device__ Array2d() : data(0), height(0), width(0) {}
  __host__ __device__ Array2d(T *data_, int width_, int height_) : data(data_), width(width_), height(height_) {}
             Array2d(thrust::device_vector<T> &vector, int width, int height); // If vector is deleted,
  __host__ __device__ Array2d(const Array2d &other) : data(other.data), width(other.width), height(other.height) {}

  int length() const { return width * height; }

  __host__ __device__ Size2d size() const { return Size2d(width, height); }

  const T* constData() const { return (const T*)data; }
  thrust::device_ptr<T> thrustPointer() { return thrust::device_ptr<T>(data); }
  thrust::device_ptr<T> thrustEndPointer() { return thrustPointer() + width*height; }

  void sort() { thrust::sort(this->thrustPointer(), this->thrustEndPointer()); }

  Array2d<T>& sorted_copy_in(Array2d<T> &output) {
    this->copyTo(output); //TODO: Check if pointers are the same... and don't copy
    output.sort();
    return output;
  }

  void copyTo(Array2d<T> &array)               { thrust::copy(thrustPointer(), thrustPointer()+width*height, array.thrustPointer()); }
  void copyTo(thrust::device_ptr<T> &array)    { thrust::copy(thrustPointer(), thrustPointer()+width*height, array); }
  void copyTo(thrust::device_vector<T> &array) { thrust::copy(thrustPointer(), thrustPointer()+width*height, array.begin()); }
  void copyTo(thrust::host_vector<T> &array)   { thrust::copy(thrustPointer(), thrustPointer()+width*height, array.begin()); }

  void swapDataWith(Array2d<T> &other) {
    T *temp = data;
    data = other.data;
    other.data = temp;
  }

  __device__
  int calculateIndex(int x, int y, int x_offset=0, int y_offset=0) const {
    x = ((x + x_offset) + width)  % width;
    y = ((y + y_offset) + height) % height;

    return y*width + x;
  }

  __device__
  T& at(int x, int y) {
    return data[calculateIndex(x, y)];
  }
};

} // namespace Walnut

#endif // ARRAY2D_H
