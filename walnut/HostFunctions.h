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

#include "Sizes.h"

#include <QString>

namespace Walnut {

WALNUT_EXPORT template <typename T>
std::string stdStringFromElement(const T &number);

WALNUT_EXPORT template <typename T>
T elementFromString(const QString &string);

WALNUT_EXPORT template <typename T>
QString stringFromElement(const T &element);

/*
  Simple debugging function to print out a 2d array.
*/
WALNUT_EXPORT template <typename T>
void printArray2D(const thrust::host_vector<T> &vector, int width, int height);

WALNUT_EXPORT template <typename T>
void printArray(const thrust::host_vector<T> &vector, const Size3d &size);

} // namespace Walnut

#endif // UTILITYFUNCTIONS_H
