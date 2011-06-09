#ifndef COLORKERNELS_H
#define COLORKERNELS_H

#include <thrust/tuple.h>

struct _chestnut_default_color_conversion_functor {
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t) {
    uchar4 color;
    color.x = thrust::get<1>(t);
    color.y = thrust::get<1>(t);
    color.z = thrust::get<1>(t);
    color.w = 255;
    thrust::get<0>(t) = color;
  }
};

struct _chestnut_red_color_conversion_functor {
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t) {
    uchar4 color;
    color.x = thrust::get<1>(t);
    color.y = 0;
    color.z = 0;
    color.w = 255;
    thrust::get<0>(t) = color;
  }
};

struct _chestnut_green_color_conversion_functor {
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t) {
    uchar4 color;
    color.x = 0;
    color.y = thrust::get<1>(t);
    color.z = 0;
    color.w = 255;
    thrust::get<0>(t) = color;
  }
};

struct _chestnut_blue_color_conversion_functor {
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t) {
    uchar4 color;
    color.x = 0;
    color.y = 0;
    color.z = thrust::get<1>(t);
    color.w = 255;
    thrust::get<0>(t) = color;
  }
};


#endif //COLORKERNELS_H
