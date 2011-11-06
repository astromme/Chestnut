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

#include "FunctionIterator.h"

namespace Walnut {

FunctionIterator makeStartIterator(int width, int height, int depth) {

  IndexIterator  indexIterator(0);
  WidthIterator  widthIterator(width);
  HeightIterator heightIterator(height);
  DepthIterator  depthIterator(depth);

  return FunctionIterator(FunctionTuple(indexIterator, widthIterator, heightIterator, depthIterator));
}

FunctionIterator makeStartIterator(const Size1d &size) {
    return makeStartIterator(size.width());
}

FunctionIterator makeStartIterator(const Size2d &size) {
    return makeStartIterator(size.width(), size.height());
}

FunctionIterator makeStartIterator(const Size3d &size) {
    return makeStartIterator(size.width(), size.height(), size.depth());
}

FunctionIterator makeEndIterator(FunctionIterator startIterator, int width, int height, int depth) {
  return startIterator + width*height*depth;
}

} // namespace Walnut
