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

#ifndef WINDOWS_H
#define WINDOWS_H

#include "walnut_global.h"
#include "Array.h"

namespace Walnut {

template <typename T>
struct WALNUT_EXPORT Window {

  Array<T> array;

  int m_xLocation;
  int m_yLocation;
  int m_zLocation;

  __device__ Window() { m_xLocation = m_yLocation = m_zLocation = 0; array = Array<T>(); }
  __device__ Window(Array<T> &data, int x, int y=0, int z=0) : array(data), m_xLocation(x), m_yLocation(y), m_zLocation(z) {}
  __device__ Window(const Window<T> &other) : array(other.array), m_xLocation(other.m_xLocation), m_yLocation(other.m_yLocation), m_zLocation(other.m_zLocation) {}

  __device__ Window<T> & operator=(const Window<T> &rhs) { this->center() = rhs.center(); return *this; }
  __device__ Window<T> & operator+=(const Window<T> &rhs) { this->center() += rhs.center(); return *this; }
  __device__ Window<T> & operator-=(const Window<T> &rhs) { this->center() -= rhs.center(); return *this; }
  __device__ Window<T> & operator*=(const Window<T> &rhs) { this->center() *= rhs.center(); return *this; }
  __device__ Window<T> & operator/=(const Window<T> &rhs) { this->center() /= rhs.center(); return *this; }
  __device__ Window<T>   operator+(const Window<T> &rhs) { Window<T> result = *this; return (result += rhs); }
  __device__ Window<T>   operator-(const Window<T> &rhs) { Window<T> result = *this; return (result -= rhs); }
  __device__ Window<T>   operator*(const Window<T> &rhs) { Window<T> result = *this; return (result *= rhs); }
  __device__ Window<T>   operator/(const Window<T> &rhs) { Window<T> result = *this; return (result /= rhs); }

  __device__ operator T() { return this->center(); }
  __device__ bool operator==(const T &rhs) { return this->center() == rhs; }
  __device__ bool operator!=(const T &rhs) { return !(*this == rhs); }
  __device__ Window<T> & operator=(const T &rhs) { this->center() = rhs; return *this; }
  __device__ Window<T> & operator+=(const T &rhs) { this->center() += rhs; return *this; }
  __device__ Window<T> & operator-=(const T &rhs) { this->center() -= rhs; return *this; }
  __device__ Window<T> & operator*=(const T &rhs) { this->center() *= rhs; return *this; }
  __device__ Window<T> & operator/=(const T &rhs) { this->center() /= rhs; return *this; }
  __device__ T   operator+(const T &rhs) { return this->center() + rhs; }
  __device__ T   operator-(const T &rhs) { return this->center() - rhs; }
  __device__ T   operator*(const T &rhs) { return this->center() / rhs; }
  __device__ T   operator/(const T &rhs) { return this->center() * rhs; }
  //TODO: operator+=, operator-=, operator*=, operator/=?

  __device__
  T& at(int x_offset=0, int y_offset=0, int z_offset=0) {
    return array.data[array.calculateIndex(m_xLocation, m_yLocation, m_zLocation, x_offset, y_offset, z_offset)];
  }

  __device__
  T at(int x_offset=0, int y_offset=0, int z_offset=0) const {
    return array.data[array.calculateIndex(m_xLocation, m_yLocation, m_zLocation, x_offset, y_offset, z_offset)];
  }

  __device__ Int x() const             { return m_xLocation; }
  __device__ Int y() const             { return m_yLocation; }
  __device__ Int z() const             { return m_zLocation; }

  __device__ T& center()               { return at(); }
  __device__ T center() const          { return at(); }
  __device__ T& left()                 { return at(-1); }
  __device__ T left() const            { return at(-1); }
  __device__ T& right()                { return at( 1); }
  __device__ T right() const           { return at( 1); }

  __device__ T& north()                { return at( 0, -1); }
  __device__ T north() const           { return at( 0, -1); }
  __device__ T& northEast()            { return at( 1, -1); }
  __device__ T northEast() const       { return at( 1, -1); }
  __device__ T& east()                 { return at( 1,  0); }
  __device__ T east() const            { return at( 1,  0); }
  __device__ T& southEast()            { return at( 1,  1); }
  __device__ T southEast() const       { return at( 1,  1); }
  __device__ T& south()                { return at( 0,  1); }
  __device__ T south() const           { return at( 0,  1); }
  __device__ T& southWest()            { return at(-1,  1); }
  __device__ T southWest() const       { return at(-1,  1); }
  __device__ T& west()                 { return at(-1,  0); }
  __device__ T west() const            { return at(-1,  0); }
  __device__ T& northWest()            { return at(-1, -1); }
  __device__ T northWest() const       { return at(-1, -1); }

  __device__ T& above()                { return at( 0,  0, -1); }
  __device__ T above() const           { return at( 0,  0, -1); }
  __device__ T& aboveNorth()           { return at( 0, -1, -1); }
  __device__ T aboveNorth() const      { return at( 0, -1, -1); }
  __device__ T& aboveNorthEast()       { return at( 1, -1, -1); }
  __device__ T aboveNorthEast() const  { return at( 1, -1, -1); }
  __device__ T& aboveEast()            { return at( 1,  0, -1); }
  __device__ T aboveEast() const       { return at( 1,  0, -1); }
  __device__ T& aboveSouthEast()       { return at( 1,  1, -1); }
  __device__ T aboveSouthEast() const  { return at( 1,  1, -1); }
  __device__ T& aboveSouth()           { return at( 0,  1, -1); }
  __device__ T aboveSouth() const      { return at( 0,  1, -1); }
  __device__ T& aboveSouthWest()       { return at(-1,  1, -1); }
  __device__ T aboveSouthWest() const  { return at(-1,  1, -1); }
  __device__ T& aboveWest()            { return at(-1,  0, -1); }
  __device__ T aboveWest() const       { return at(-1,  0, -1); }
  __device__ T& aboveNorthWest()       { return at(-1, -1, -1); }
  __device__ T aboveNorthWest() const  { return at(-1, -1, -1); }

  __device__ T& below()                { return at( 0,  0, 1); }
  __device__ T below() const           { return at( 0,  0, 1); }
  __device__ T& belowNorth()           { return at( 0, -1, 1); }
  __device__ T belowNorth() const      { return at( 0, -1, 1); }
  __device__ T& belowNorthEast()       { return at( 1, -1, 1); }
  __device__ T belowNorthEast() const  { return at( 1, -1, 1); }
  __device__ T& belowEast()            { return at( 1,  0, 1); }
  __device__ T belowEast() const       { return at( 1,  0, 1); }
  __device__ T& belowSouthEast()       { return at( 1,  1, 1); }
  __device__ T belowSouthEast() const  { return at( 1,  1, 1); }
  __device__ T& belowSouth()           { return at( 0,  1, 1); }
  __device__ T belowSouth() const      { return at( 0,  1, 1); }
  __device__ T& belowSouthWest()       { return at(-1,  1, 1); }
  __device__ T belowSouthWest() const  { return at(-1,  1, 1); }
  __device__ T& belowWest()            { return at(-1,  0, 1); }
  __device__ T belowWest() const       { return at(-1,  0, 1); }
  __device__ T& belowNorthWest()       { return at(-1, -1, 1); }
  __device__ T belowNorthWest() const  { return at(-1, -1, 1); }

};

typedef Window<Int> Int1d;
typedef Window<Int> Int2d;
typedef Window<Int> Int3d;

typedef Window<Real> Real1d;
typedef Window<Real> Real2d;
typedef Window<Real> Real3d;

typedef Window<Color> Color1d;
typedef Window<Color> Color2d;
typedef Window<Color> Color3d;

typedef Window<Bool> Bool1d;
typedef Window<Bool> Bool2d;
typedef Window<Bool> Bool3d;

}

#endif // WINDOWS_H
