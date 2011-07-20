#ifndef COLORKERNELS_H
#define COLORKERNELS_H

#include "walnut_global.h"

#include <thrust/tuple.h>

namespace Walnut {

template <typename InputType>
struct WALNUT_EXPORT _chestnut_default_color_conversion_functor : public thrust::unary_function<InputType, color> {

  __device__
  int convert(float value) {
    return wBound(0, int(value * 255), 255);
  }

  __device__
  int convert(int value) {
    return wBound(0, value, 255);
  }

  __device__
  color operator()(InputType value) {
    color gray;
    gray.x = convert(value);
    gray.y = convert(value);
    gray.z = convert(value);
    gray.w = 255;
    return gray;
  }
};

} // end namespace Walnut

#endif //COLORKERNELS_H
