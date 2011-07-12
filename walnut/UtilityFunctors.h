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

//struct make_random_float2
//{
//};
//__host__ __device__ float2 operator()(int index) {
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
//__host__ __device__ float2 operator()(int index) {
//return make_float2( (float)hash(2*index + 0) / UINT_MAX, (float)hash(2*index + 1) / UINT_MAX);
//}
//// generate random input directly on the device
//device_vector<float2> points(N); transform(make_counting_iterator(0), make_counting_iterator(N),
//points.begin(), make_random_float2());


///*
//* This program uses the device CURAND API to calculate what
//* proportion of pseudo - random ints have low bit set .
//*/
//# include < stdio .h >
//# include < stdlib .h >
//# include < cuda .h >
//# include < curand_kernel .h >
//# define CUDA_CALL ( x) do { if (( x) != cudaSuccess ) { \
//printf (" Error at % s :% d\ n" , __FILE__ , __LINE__ ); \
//return EXIT_FAILURE ;}} while (0)
//__global__ void setup_kernel ( curandState * state )
//{
//int id = threadIdx . x + blockIdx .x * 64;
///* Each thread gets same seed , a different sequence number  -
//,
//no offset */
//curand_init (1234 , id , 0 , & state [ id ]) ;
//}
//__global__ void generate_kernel ( curandState * state , int * -
//result )
//{
//int id = threadIdx . x + blockIdx .x * 64;
//int count = 0;
//unsigned int x ;
///* Copy state to local memory for efficiency */
//CUDA Toolkit 4.0 CURAND Guide PG-05328-040_v01 | 16curandState localState = state [ id ];
///* Generate pseudo - random unsigned ints */
//for ( int n = 0; n < 100000; n ++) {
//x = curand (& localState ) ;
///* Check if low bit set */
//if (x & 1) {
//count ++;
//}
//}
///* Copy state back to global memory */
//state [ id ] = localState ;
///* Store results */
//result [ id ] += count ;
//}
//int main ( int argc , char * argv [])
//{
//int i , total ;
//curandState * devStates ;
//int * devResults , * hostResults ;
///* Allocate space for results on host */
//hostResults = ( int *) calloc (64 * 64 , sizeof ( int )) ;
///* Allocate space for results on device */
//CUDA_CALL ( cudaMalloc (( void **) & devResults , 64 * 64 *  -
//sizeof ( int )) );
///* Set results to 0 */
//CUDA_CALL ( cudaMemset ( devResults , 0, 64 * 64 * sizeof ( int ) ) -
//);
///* Allocate space for prng states on device */
//CUDA_CALL ( cudaMalloc (( void **) & devStates , 64 * 64 *
//sizeof ( curandState ) ));
///* Setup prng states */
//setup_kernel < < <64 , 64 > > >( devStates ) ;
///* Generate and use pseudo - random */
//for ( i = 0; i < 10; i ++) {
//generate_kernel < < <64 , 64 > > >( devStates , devResults ) ;
//}
//CUDA Toolkit 4.0 CURAND Guide PG-05328-040_v01 | 17/* Copy device memory to host */
//CUDA_CALL ( cudaMemcpy ( hostResults , devResults , 64 * 64 *
//sizeof ( int ) , cudaMemcpyDeviceToHost )) ;
///* Show result */
//total = 0;
//for ( i = 0; i < 64 * 64; i ++) {
//total += hostResults [i ];
//}
//printf (" Fraction with low bit set was %10.13 f\ n" ,
//( float ) total / (64.0 f * 64.0 f * 100000.0 f * 10.0 f) );
///* Cleanup */
//CUDA_CALL ( cudaFree ( devStates )) ;
//CUDA_CALL ( cudaFree ( devResults ) );
//free ( hostResults ) ;
//return EXIT_SUCCESS ;


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
