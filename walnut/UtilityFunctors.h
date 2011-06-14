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

#ifndef UTILITYFUNCTORS_H
#define UTILITYFUNCTORS_H

#include "walnut_global.h"
#include <thrust/transform.h>

namespace Walnut {



//// A CPU-driven random function that fills the data
//void randomize(int minValue, int maxValue) {
//  // Create host vector to hold the random numbers
//  thrust::host_vector<T> randoms(width*height);

//  // Generate the random numbers
//  thrust::generate(randoms.begin(), randoms.end(), rand);

//  // Copy data from CPU to GPU
//  *mainData = randoms;

//  thrust::for_each(mainData->begin(),
//                   mainData->end(),
//                   randoms_helper_functor(minValue, maxValue));
//}

//void randomize(int maxValue=INT_MAX) {
//  randomize(0, maxValue);
//}

struct WALNUT_EXPORT randoms_helper_functor {
  int minValue;
  int maxValue;

  randoms_helper_functor(int minValue_, int maxValue_) : minValue(minValue_), maxValue(maxValue_) {}

  __host__ __device__
  void operator()(int &value)
  {
    value = (value % (maxValue - minValue)) + minValue;
  }
};

} // namespace Walnut

#endif // UTILITYFUNCTORS_H