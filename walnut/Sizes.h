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

struct Size1
{
  int32 width;
  Size1(int32 width_) : width(width_) {}
  int32 length() const { return width; }
};

struct Size2 {
  int32 width;
  int32 height;
  Size2(int32 width_, int32 height_) : width(width_), height(height_) {}
  int32 length() const { return width * height; }

};

struct Size3 {
  int32 width;
  int32 height;
  int32 depth;
  Size3(int32 width_, int32 height_, int32 depth_) : width(width_), height(height_), depth(depth_) {}
  int32 length() const { return width * height * depth; }
};

} // namespace Walnut

#endif // SIZES_H