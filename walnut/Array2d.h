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
#include <thrust/device_vector.h>

namespace Walnut {

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

  Array2d(T *data, int width, int height);
  Array2d(thrust::device_vector<T> &vector, int width, int height);

  const T* constData() const { return (const T*)data; }
  thrust::device_ptr<T> thrustPointer() { return thrust::device_ptr<T>(data); }
  thrust::device_ptr<T> thrustEndPointer() { return thrustPointer() + width*height; }

  void copyTo(Array2d<T> &array)               { thrust::copy(thrustPointer(), thrustPointer()+width*height, array.thrustPointer()); }
  void copyTo(thrust::device_ptr<T> &array)    { thrust::copy(thrustPointer(), thrustPointer()+width*height, array); }
  void copyTo(thrust::device_vector<T> &array) { thrust::copy(thrustPointer(), thrustPointer()+width*height, array.begin()); }
  void copyTo(thrust::host_vector<T> &array)   { thrust::copy(thrustPointer(), thrustPointer()+width*height, array.begin()); }

  void swapDataWith(Array2d<T> &other) {
    T *temp = data;
    data = other.data;
    other.data = temp;
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

  __host__ __device__ T topLeft(int x, int y) const     { return shiftedData(x, y, -1, -1); }
  __host__ __device__ T top(int x, int y) const         { return shiftedData(x, y,  0, -1); }
  __host__ __device__ T topRight(int x, int y) const    { return shiftedData(x, y,  1, -1); }
  __host__ __device__ T left(int x, int y) const        { return shiftedData(x, y, -1,  0); }
  __host__ __device__ T center(int x, int y) const      { return shiftedData(x, y,  0,  0); }
  __host__ __device__ T right(int x, int y) const       { return shiftedData(x, y,  1,  0); }
  __host__ __device__ T bottomLeft(int x, int y) const  { return shiftedData(x, y, -1,  1); }
  __host__ __device__ T bottom(int x, int y) const      { return shiftedData(x, y,  0,  1); }
  __host__ __device__ T bottomRight(int x, int y) const { return shiftedData(x, y,  1,  1); }

};

} // namespace Walnut

#endif // ARRAY2D_H
