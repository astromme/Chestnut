#ifndef COLORKERNELS_H
#define COLORKERNELS_H

#include "walnut_global.h"

#include <thrust/tuple.h>

namespace Walnut {

template <typename InputType>
struct WALNUT_EXPORT _chestnut_default_color_conversion_functor : public thrust::unary_function<InputType, color> {
  __host__ __device__
  color operator()(int32 input) {
    color gray;
    gray.x = wBound(0, input, 255);
    gray.y = wBound(0, input, 255);
    gray.z = wBound(0, input, 255);
    gray.w = 255;
    return gray;
  }

  __host__ __device__
  color operator()(real32 input) {
    color gray;
    gray.x = wBound(0, int(input*255), 255);
    gray.y = wBound(0, int(input*255), 255);
    gray.z = wBound(0, int(input*255), 255);
    gray.w = 255;
    return gray;
  }
};

} // end namespace Walnut

#endif //COLORKERNELS_H
