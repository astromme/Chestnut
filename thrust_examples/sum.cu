#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <cstdlib>
#include <cstdio>

int main(void)
{
  int numvals = 10000000;
  // generate random data on the host
  thrust::host_vector<int> h_vec(numvals);
  for (int i=0; i<numvals; i++){
    h_vec[i] = 1;
  }
  //thrust::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer to device and compute sum
  thrust::device_vector<int> d_vec = h_vec;
  int x = thrust::reduce(d_vec.begin(), d_vec.end(), (int) 0,
                          thrust::plus<int>());

  printf("Done: %d\n", x);
  return 0;
}
