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

#ifndef ARRAY_H
#define ARRAY_H

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
struct WALNUT_EXPORT Array
{
  T *data;
  int width;
  int height;
  int depth;

  typedef T Type;

  __host__ __device__ Array() : data(0), height(0), width(0), depth(0) {}
  __host__ __device__ Array(T *data_, int width_, int height_=1, int depth_=1) : data(data_), width(width_), height(height_), depth(depth_) {}
             Array(thrust::device_vector<T> &vector, int width, int height, int depth); // If vector is deleted, bad stuff happens
             __host__ __device__ Array(const Array &other) : data(other.data), width(other.width), height(other.height), depth(other.depth) {}

  int length() const { return width * height * depth; }

  __host__ __device__ Size3d size() const { return Size3d(width, height, depth); }

  const T* constData() const { return (const T*)data; }
  thrust::device_ptr<T> thrustPointer() { return thrust::device_ptr<T>(data); }
  thrust::device_ptr<T> thrustEndPointer() { return thrustPointer() + width*height*depth; }

  void sort() { thrust::sort(this->thrustPointer(), this->thrustEndPointer()); }

  Array<T>& sorted_copy_in(Array<T> &output) {
    this->copyTo(output); //TODO: Check if pointers are the same... and don't copy
    output.sort();
    return output;
  }

  void copyTo(Array<T> &array)               { thrust::copy(thrustPointer(), thrustPointer()+width*height*depth, array.thrustPointer()); }
  void copyTo(thrust::device_ptr<T> &array)    { thrust::copy(thrustPointer(), thrustPointer()+width*height*depth, array); }
  void copyTo(thrust::device_vector<T> &array) { thrust::copy(thrustPointer(), thrustPointer()+width*height*depth, array.begin()); }
  void copyTo(thrust::host_vector<T> &array)   { thrust::copy(thrustPointer(), thrustPointer()+width*height*depth, array.begin()); }

  void swapDataWith(Array<T> &other) {
    T *temp = data;
    data = other.data;
    other.data = temp;
  }

  __device__
  int calculateIndex(int x, int y, int z, int x_offset=0, int y_offset=0, int z_offset=0) const {
    x = ((x + x_offset) + width)  % width;
    y = ((y + y_offset) + height) % height;
    z = ((z + z_offset) + depth) % depth;

    return z*width*height + y*width + x;
  }

  __device__
  T& at(int x, int y=0, int z=0) {
    return data[calculateIndex(x, y, z)];
  }
};

} // namespace Walnut

#endif // ARRAY_H
