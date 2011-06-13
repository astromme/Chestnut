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

#ifndef UTILITYFUNCTIONS_H
#define UTILITYFUNCTIONS_H

#include "walnut_global.h"

#include <iostream>
#include <sstream>

#include <thrust/host_vector.h>

namespace Walnut {

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


} // namespace Walnut

#endif // UTILITYFUNCTIONS_H
