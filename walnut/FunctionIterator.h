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

#ifndef FUNCTIONITERATOR_H
#define FUNCTIONITERATOR_H

#include <thrust/tuple.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

namespace Walnut {

// constant iterator information for the location in the array + the bounds
// from these we can compute the x, y, width, height inside of the kernel
typedef thrust::counting_iterator<int> IndexIterator;

typedef thrust::constant_iterator<int> WidthIterator;
typedef thrust::constant_iterator<int> HeightIterator;

typedef thrust::tuple<IndexIterator, WidthIterator, HeightIterator> FunctionTuple;

// zipped iterator for the above window tuple
typedef thrust::zip_iterator<FunctionTuple> FunctionIterator;

FunctionIterator makeStartIterator(int width, int height);
FunctionIterator makeEndIterator(int width, int height, FunctionIterator startIterator);

} // namespace Walnut

#endif // FUNCTIONITERATOR_H
