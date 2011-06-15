#ifndef COLORKERNELS_H
#define COLORKERNELS_H

#include "walnut_global.h"

#include <thrust/tuple.h>

namespace Walnut {

template <typename OutputType, typename InputType>
struct WALNUT_EXPORT _chestnut_default_color_conversion_functor : public thrust::unary_function<InputType, OutputType> {

  Array2d<InputType> input;
  _chestnut_default_color_conversion_functor(Array2d<InputType> &input_) : input(input_) {}

  __host__ __device__
  int convert(float value) {
    return wBound(0, int(value * 255), 255);
  }

  __host__ __device__
  int convert(int value) {
    return wBound(0, value, 255);
  }

  template <typename Tuple>
  __host__ __device__
  OutputType operator()(Tuple _t) {
    color gray;
    gray.x = convert(input.center(_x, _y));
    gray.y = convert(input.center(_x, _y));
    gray.z = convert(input.center(_x, _y));
    gray.w = 255;
    return gray;
  }
};

} // end namespace Walnut

#endif //COLORKERNELS_H
