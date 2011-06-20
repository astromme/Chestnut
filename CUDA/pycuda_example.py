#!/usr/bin/env python

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

four_by_four_array = numpy.random.randn(4,4)
four_by_four_array = four_by_four_array.astype(numpy.float32)

gpu_array = cuda.mem_alloc(four_by_four_array.nbytes)
cuda.memcpy_htod(gpu_array, four_by_four_array)

mod = SourceModule("""
  __global__ void doublify(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
  }
  """)


func = mod.get_function("doublify")
func(gpu_array, block=(4,4,1))

doubled = numpy.empty_like(four_by_four_array)
cuda.memcpy_dtoh(doubled, gpu_array)
print doubled
print four_by_four_array


