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

#include "walnut_global.h"

namespace Walnut {

__host__ __device__
complex::complex(real32 real_part_, real32 imaginary_part_)
 : real_part(real_part_), imaginary_part(imaginary_part_)
{
}

__host__ __device__
real32 complex::magnitude() const {
  return sqrtf(magnitudeSquared());
}

__host__ __device__
real32 complex::magnitudeSquared() const {
  return (real_part*real_part) + (imaginary_part*imaginary_part);
}

__host__ __device__
complex complex::operator*(const complex &other) const {
  return complex(real_part*other.real_part - imaginary_part*other.imaginary_part,
                 imaginary_part*other.real_part + real_part*other.imaginary_part);
}

__host__ __device__
complex complex::operator+(const complex &other) const {
  return complex(real_part+other.real_part, imaginary_part+other.imaginary_part);
}


} // namespace Walnut
