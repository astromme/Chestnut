#ifndef _PARSE_LIBS_H
#define _PARSE_LIBS_H

template <typename TYPE>
class DataInfo {
  public:
    int rows;
    int cols;
    TYPE* data;
};

#endif
