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

#ifndef COLOR_H
#define COLOR_H

#include "walnut_global.h"

namespace Walnut {

struct Color : uchar4 {
  __host__ __device__ inline uint8& blue() { return this->x; }
  __host__ __device__ inline uint8& green() { return this->y; }
  __host__ __device__ inline uint8& red() { return this->z; }
  __host__ __device__ inline uint8& opacity() { return this->w; }

  __host__ __device__ inline uint8 blue() const { return this->x; }
  __host__ __device__ inline uint8 green() const { return this->y; }
  __host__ __device__ inline uint8 red() const { return this->z; }
  __host__ __device__ inline uint8 opacity() const { return this->w; }
};

}
#endif // COLOR_H
