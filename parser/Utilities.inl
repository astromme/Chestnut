/*#include <iostream>
#include <fstream>
#include "Utilities.h"

using namespace std;

// TODO: add error checking
template <typename TYPE>
TYPE* Utilities::readDatafile(std::string fname, int *myrows, int *mycols){
  ifstream fptr(fname.c_str());
  string garbage;
  int dimensions, rows, cols = -1;
  TYPE **data;
  
  fptr >> garbage; fptr >> dimensions;
  fptr >> garbage; fptr >> rows;

  if (dimensions == 1){
    TYPE* tmpdata = new TYPE[rows];

    // FIXME!!!
    //for (int r=0; r<rows; r++){
    //  fptr >> data[r];
    //}
    data = (TYPE**)tmpdata;

  } else if (dimensions == 2){
    fptr >> garbage; fptr >> cols;

    TYPE **data = new TYPE*[rows];
    for (int r=0; r<rows; r++){
      data[r] = new TYPE[cols];
    }

    //for (int r=0; r<rows; r++){
    //for (int c=0; c<cols; c++){
    //  fptr >> data[r][c];
    //}
  //}
  }
  
  *myrows = rows;
  *mycols = cols;
  return (TYPE*)data;
}*/
