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
#include <curand_kernel.h>
#include <curand.h>

#define walnut_random() Walnut::random(&__random_state)

namespace Walnut {

#if __CUDA_ARCH__ > 0
float __device__ random(curandState *state) {
 return curand_uniform(state);
}
#else
float __host__ random(curandState *state) {
 Q_UNUSED(state)
 return float(rand())/INT_MAX;
}
#endif

inline void check_cuda_errors(const char *filename, const int line_number)
{
#ifdef DEBUG
  cudaThreadSynchronize();
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
    exit(-1);
  }
#endif
}


__device__ unsigned int hash(unsigned int a) {
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

int __device__ indexFromDimensions(dim3 blockIdx, dim3 threadIdx, dim3 gridDim, dim3 blockDim) {
  int index = threadIdx.x +
              threadIdx.y * blockDim.x +
              threadIdx.z * blockDim.x * blockDim.y +
              blockIdx.x  * blockDim.x * blockDim.y * blockDim.z +
              blockIdx.y  * blockDim.x * blockDim.y * blockDim.z * gridDim.x +
              blockIdx.z  * blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y;

  return index;
}

// X access
int __device__ xFromIndex(int index, int width) {
  Q_UNUSED(width);
  return index;
}
int __device__ xFromIndex(int index, int width, int height) {
  Q_UNUSED(height);
  return index % width;
}
int __device__ xFromIndex(int index, int width, int height, int depth) {
  Q_UNUSED(height);
  Q_UNUSED(depth);
  return index % width;
}

// Y access
int __device__ yFromIndex(int index, int width, int height) {
  Q_UNUSED(height);
  return index / width;
}
int __device__ yFromIndex(int index, int width, int height, int depth) {
  Q_UNUSED(depth);
  return (index / width) % height;
}

// Z access
int __device__ zFromIndex(int index, int width, int height, int depth) {
  Q_UNUSED(depth);
  return index / (width * height);
}

__global__ void initialize_random_number_generator(curandState *state)
{
  int index = indexFromDimensions(blockIdx, threadIdx, gridDim, blockDim);

  // Each thread gets same seed , a different sequence number, no offset
  curand_init(1234 /* direction vectors */, index, 0, &state[index]);
}

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

//struct make_random_float2
//{
//};
//__device__ float2 operator()(int index) {
//}
//default_random_engine rng;
//// skip past numbers used in previous threads
//rng.discard(2*index); return make_float2( (float)rng() / default_random_engine::max,
//(float)rng() / default_random_engine::max);
//// generate random input directly on the device
//device_vector<float2> points(N); transform(make_counting_iterator(0), make_counting_iterator(N),
//points.begin(), make_random_float2());

//On the device with an integer hash: struct make_random_float2
//{
//};
//__device__ float2 operator()(int index) {
//return make_float2( (float)hash(2*index + 0) / UINT_MAX, (float)hash(2*index + 1) / UINT_MAX);
//}
//// generate random input directly on the device
//device_vector<float2> points(N); transform(make_counting_iterator(0), make_counting_iterator(N),
//points.begin(), make_random_float2());

//int main ( int argc , char * argv [])
//{
//int i, total;
//curandState *devStates;

///* Allocate space for prng states on device */
//CUDA_CALL ( cudaMalloc ((void **) & devStates , 64 * 64 *
//sizeof ( curandState ) ));

//setup_kernel<<<64, 64>>>(devStates) ;

///* Setup prng states */
///* Generate and use pseudo - random */

} // namespace Walnut

#endif // UTILITYFUNCTORS_H
