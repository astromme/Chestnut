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

#ifndef SIZES_H
#define SIZES_H

#include "walnut_global.h"

namespace Walnut {

struct Size3d {
  int32 m_width;
  int32 m_height;
  int32 m_depth;
  __host__ __device__ Size3d() : m_width(0), m_height(0), m_depth(0) {}
  //__host__ __device__ Size3d(const Size1d &size) : m_width(size.width()), m_height(1), m_depth(1) {}
  //__host__ __device__ Size3d(const Size2d &size) : m_width(size.width()), m_height(size.height()), m_depth(1) {}
  __host__ __device__ Size3d(int32 width_, int32 height_, int32 depth_) : m_width(width_), m_height(height_), m_depth(depth_) {}
  __host__ __device__ inline int32 length() const { return m_width * m_height * m_depth; }

  __host__ __device__ inline int32& width() { return m_width; }
  __host__ __device__ inline int32& height() { return m_height; }
  __host__ __device__ inline int32& depth() { return m_depth; }

  __host__ __device__ inline int32 width() const { return m_width; }
  __host__ __device__ inline int32 height() const { return m_height; }
  __host__ __device__ inline int32 depth() const { return m_depth; }
};

struct Size2d {
  int32 m_width;
  int32 m_height;
  __host__ __device__ Size2d(const Size3d &other) { m_width = other.width(); m_height = other.height(); }
  __host__ __device__ Size2d(int32 width_, int32 height_) : m_width(width_), m_height(height_) {}
  __host__ __device__ inline int32 length() const { return m_width * m_height; }

  __host__ __device__ inline int32& width() { return m_width; }
  __host__ __device__ inline int32& height() { return m_height; }

  __host__ __device__ inline int32 width() const { return m_width; }
  __host__ __device__ inline int32 height() const { return m_height; }

};


struct Size1d
{
  int32 m_width;
  __host__ __device__ Size1d(const Size3d &other) { m_width = other.width(); }
  __host__ __device__ Size1d(const Size2d &other) { m_width = other.width(); }
  __host__ __device__ Size1d(int32 width) : m_width(width) {}

  __host__ __device__ inline int32& width() { return m_width; }
  __host__ __device__ inline int32 width() const { return m_width; }

  int32 length() const { return m_width; }
};


} // namespace Walnut

#endif // SIZES_H
