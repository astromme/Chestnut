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

#include "HostFunctions.h"

namespace Walnut {

std::string stringFromInt(int number)
{
   std::stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}

/*
  Simple debugging function to print out a 2d array. Does not know about
  padding, so send it the padded width/height rather than the base width/height
*/
template <typename T>
void printFullArray2D(const thrust::host_vector<T> &vector, int width, int height) {
  for (int y=0; y<height; y++) {
    for (int x=0; x<width; x++) {
      int i = y*width + x;
      std::cout << ((vector[i] == 0) ? "." : stringFromInt(vector[i])) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

/*
   Slightly smarter padding-aware print function. Send it in the width and the
   padding and it will do the dirty work for you.
*/
template <typename T>
void printArray2D(const thrust::host_vector<T> &vector, int width, int height, int padding) {
  int paddedWidth = width + 2*padding;
  //int paddedHeight = height + 2*padding;

  for (int y=padding; y<height+padding; y++) {
    for (int x=padding; x<width+padding; x++) {
      int i = y*paddedWidth + x;
      std::cout << ((vector[i] == 0) ? "." : stringFromInt(vector[i])) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

#define initPrintFunctionsWithType(T) \
template void printFullArray2D(const thrust::host_vector<T> &vector, int width, int height); \
template void printArray2D(const thrust::host_vector<T> &vector, int width, int height, int padding); \

initPrintFunctionsWithType(int);
initPrintFunctionsWithType(char);
initPrintFunctionsWithType(float);

} // namespace Walnut
