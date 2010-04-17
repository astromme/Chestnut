#ifndef _PARSE_LIBS_H
#define _PARSE_LIBS_H

// include all the boost libraries

template <typename TYPE>
class DataInfo {
  public:
    int rows;
    int cols;
    TYPE* data;
};

template <typename TYPE>
TYPE* copyData(TYPE* source, int rows, int cols);

#include "ParseLibs.inl"

#endif
