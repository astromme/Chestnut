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

#include <iostream>
#include <fstream>
#include <sstream>

__device__ inline bool operator<(const Walnut::Color &left, const Walnut::Color &right) {
  return left.red() + left.green() + left.blue() + left.opacity() < right.red() + right.green() + right.blue() + right.opacity();
}

namespace Walnut {

#define _index thrust::get<0>(_t)
#define _width thrust::get<1>(_t)
#define _height thrust::get<2>(_t)
#define _depth thrust::get<3>(_t)
#define _x (_index % _width)
#define _y ((_index / _width) % _height)
#define _z (_index / (_width * _height))

template <typename T>
struct WALNUT_EXPORT Array
{
  T *data;
  int m_width;
  int m_height;
  int m_depth;

  typedef T Type;

  __host__ __device__ Array() : data(0), m_width(0), m_height(0), m_depth(0) {}
  __host__ __device__ Array(const Array &other) : data(other.data), m_width(other.m_width), m_height(other.m_height), m_depth(other.m_depth) {}
  __host__ __device__ Array(T *data_, int width, int height=1, int depth=1) : data(data_), m_width(width), m_height(height), m_depth(depth) {}
             Array(thrust::device_vector<T> &vector, int width, int height, int depth); // If vector is deleted, bad stuff happens

  Array<T>& readFromFile(const QString &fileName);
  Array<T>& writeToFile(const QString &fileName);

  int length() const { return width() * height() * depth(); }

  __host__ __device__ Size3d size() const { return Size3d(width(), height(), depth()); }

  const T* constData() const { return (const T*)data; }

  thrust::device_ptr<T> thrustPointer() { return thrust::device_ptr<T>(data); }
  thrust::device_ptr<T> thrustPointer() const { return thrust::device_ptr<T>(data); }

  thrust::device_ptr<T> thrustEndPointer() { return thrustPointer() + width()*height()*depth(); }
  thrust::device_ptr<T> thrustEndPointer() const { return thrustPointer() + width()*height()*depth(); }


  void sort() { thrust::sort(this->thrustPointer(), this->thrustEndPointer()); }

  Array<T>& sorted_copy_in(Array<T> &output) {
    this->copyTo(output); //TODO: Check if pointers are the same... and don't copy
    output.sort();
    return output;
  }

  __host__ __device__ __inline__ int width()  const { return m_width; }
  __host__ __device__ __inline__ int height() const { return m_height; }
  __host__ __device__ __inline__ int depth()  const { return m_depth; }

  void copyTo(Array<T> &array)                 const { thrust::copy(thrustPointer(), thrustPointer()+width()*height()*depth(), array.thrustPointer()); }
  void copyTo(thrust::device_ptr<T> &array)    const { thrust::copy(thrustPointer(), thrustPointer()+width()*height()*depth(), array); }
  void copyTo(thrust::device_vector<T> &array) const { thrust::copy(thrustPointer(), thrustPointer()+width()*height()*depth(), array.begin()); }
  void copyTo(thrust::host_vector<T> &array)   const { thrust::copy(thrustPointer(), thrustPointer()+width()*height()*depth(), array.begin()); }

  // note... swapping arrays means that any copies of this Array<T> are invalid.
  void swapDataWith(Array<T> &other) {
    T *temp = data;
    data = other.data;
    other.data = temp;
  }

  __host__ __device__
  int calculateIndex(int x, int y, int z, int x_offset=0, int y_offset=0, int z_offset=0) const {
    x = ((x + x_offset) + width())  % width();
    y = ((y + y_offset) + height()) % height();
    z = ((z + z_offset) + depth()) % depth();

    return z*width()*height() + y*width() + x;
  }

  __device__
  T& at(int x, int y=0, int z=0) {
    return data[calculateIndex(x, y, z)];
  }

 
  friend __host__ std::ostream& operator<<(std::ostream& os, const Array<T>& obj)
  {
    // Create host vector to hold data
    thrust::host_vector<T> hostArray(obj.size().length());

    // transfer data back to host
    obj.copyTo(hostArray);

    //TODO: Convert this to something that reutrns a string, then we're good.
    for (int z=0; z<obj.size().depth(); z++) {
        for (int y=0; y<obj.size().height(); y++) {
            for (int x=0; x<obj.size().width(); x++) {
                int i = z*(obj.size().width()*obj.size().height()) + y*obj.size().width() + x;
                os << ((hostArray[i] == 0) ? "." : stdStringFromElement(hostArray[i])) << " ";
             }
             os << std::endl;
        }
        os << std::endl;
    }

    return os;
  }  

};

typedef Array<Int> IntArray1d;
typedef Array<Int> IntArray2d;
typedef Array<Int> IntArray3d;

typedef Array<Real> RealArray1d;
typedef Array<Real> RealArray2d;
typedef Array<Real> RealArray3d;

typedef Array<Color> ColorArray1d;
typedef Array<Color> ColorArray2d;
typedef Array<Color> ColorArray3d;

typedef Array<Bool> BoolArray1d;
typedef Array<Bool> BoolArray2d;
typedef Array<Bool> BoolArray3d;

} // namespace Walnut

#endif // ARRAY_H
