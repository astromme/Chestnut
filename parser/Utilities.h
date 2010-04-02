#ifndef _UTILITIES_H
#define _UTILITIES_H

#include <string>

class Utilities {
  public:
    Utilities();
   
    template <typename TYPE> TYPE* readDatafile(std::string fname, int *rows, int *cols);
};

#include "Utilities.inl"

#endif
