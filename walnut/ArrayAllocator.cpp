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

#include <cuda.h>

namespace Walnut {

ArrayAllocator::ArrayAllocator()
{
}

ArrayAllocator::~ArrayAllocator() {
  foreach(DeviceMemoryPointer pointer, m_freeMemory.values()) {
    cudaFree(pointer);
  }
  foreach(DeviceMemoryPointer pointer, m_usedMemory.keys()) {
    qWarning("Freeing memory that is marked as in use");
    cudaFree(pointer);
  }
}

template <typename T>
Array<T> ArrayAllocator::arrayWithSize(int width, int height, int depth) {
  ArrayLengthInBytes arrayLength = width*height*depth*sizeof(T);
  DeviceMemoryPointer dataPointer = 0;

  // Ok, do the dumb thing, if there isn't already a free entry of this size, create a new one
  if ( m_freeMemory.values(arrayLength).length()) { // Using existing memory
    dataPointer = m_freeMemory.values(arrayLength).first();
    m_freeMemory.remove(arrayLength, dataPointer);
  } else { // Creating new memory
    cudaMalloc((void**)(&dataPointer), arrayLength);
  }

  m_usedMemory.insert(dataPointer, arrayLength);

  return Array<T>((T*)dataPointer, width, height, depth);
}

template <typename T>
Array<T> ArrayAllocator::arrayWithSize(const Size3d &size) {
    return arrayWithSize<T>(size.width(), size.height(), size.depth());
}

template <typename T>
void ArrayAllocator::releaseArray(const Array<T> &array) {
  ArrayLengthInBytes arrayLength = m_usedMemory.take((DeviceMemoryPointer)(array.constData()));
  m_freeMemory.insert(arrayLength, (DeviceMemoryPointer)(array.constData()));
}

} // namespace Walnut
