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

#ifndef ARRAYALLOCATOR_H
#define ARRAYALLOCATOR_H

#include "walnut_global.h"
#include "Array.h"
#include "Sizes.h"

#include <QMap>
#include <QMultiMap>


typedef void* DeviceMemoryPointer;
typedef void* HostMemoryPointer;
typedef size_t ArrayLengthInBytes;

namespace Walnut {

class WALNUT_EXPORT ArrayAllocator
{
public:
    ArrayAllocator();
    virtual ~ArrayAllocator();

    template <typename T>
    Array<T> arrayWithSize(int width, int height=1, int depth=1);

    template <typename T>
    Array<T> arrayWithSize(const Size3d &size);

    template <typename T>
    void releaseArray(const Array<T> &array);

private:
    QMultiMap<ArrayLengthInBytes, DeviceMemoryPointer> m_freeMemory;
    QMap<DeviceMemoryPointer, ArrayLengthInBytes> m_usedMemory;
};

} // namespace Walnut

#include "ArrayAllocator.cpp"

#endif // ARRAYALLOCATOR_H

