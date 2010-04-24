#include "datautils.h"

const ProgramStrings operator+(const ProgramStrings& p1, const ProgramStrings& p2) {
  return ProgramStrings(p1.first + p2.first, p1.second + p2.second);
}



 