#ifndef _PARSE_LIBS_H
#define _PARSE_LIBS_H

#include <string>
// Thrust includes
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using thrust::host_vector;
using std::string;

template <typename TYPE>
class DataInfo {
  public:
    int rows;
    int cols;
    host_vector<TYPE> data;
};

template <typename TYPE>
void writeDatafile(string fname, thrust::host_vector<TYPE> data, int rows, int cols);

template <typename TYPE>
TYPE* copyData(TYPE* source, int rows, int cols);

#include "ParseLibs.inl"

#endif
