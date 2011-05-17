#include <stdexcept>
#include <fstream>
#include <thrust/host_vector.h>
#include "ParseLibs.h"

using std::string;

/*********************************
 * Function: writeDatafile
 * -----------------------
 * Writes data from a thrust::host_vector out to disk
 *
 * Inputs:
 *    fname: filename we write to
 *    data: data to write to disk
 *    rows, cols: dimensions of data
 */
template <typename TYPE>
void writeDatafile(string fname, thrust::host_vector<TYPE> data, int rows, int cols){
  // output data to file
  std::ofstream outstream;
  outstream.open(fname.c_str());

  // write data out to file
  outstream << "rows: " << rows << "\n";
  outstream << "cols: " << cols << "\n";
  for (int r=0; r<rows; r++){
    for (int c=0; c<cols; c++){
      outstream << data[r*cols+c] << " ";
    }
    outstream << "\n";
  }
  outstream.close();
}

/*********************************
 * Function: readDatafile
 * -----------------------
 * Reads data in from disk to a thrust::host_vector
 *
 * Inputs:
 *    fname: file we read from
 *
 * Returns:
 *    DataInfo struct that packages the data object along with its dimensions
 */
template <typename TYPE>
DataInfo<TYPE>* readDatafile(char* fname){
  std::ifstream instream;
  instream.open(fname);

  // check that file exists
  if (instream.fail()){
    char error_msg[100+string(fname).length()];
    sprintf(error_msg, "in readDatafile: '%s' does not exist", fname);
    throw std::runtime_error(error_msg);
  }

  // read in rows and column data
  string garbage;
  int rows, cols;
  instream >> garbage; instream >> rows;
  instream >> garbage; instream >> cols;

  // read in data from file
  thrust::host_vector<TYPE> data(rows*cols);
  for (int r=0; r<rows; r++){
    for (int c=0; c<cols; c++){
      instream >> data[r*cols+c];
    }
  }
  instream.close();

  // package data information in DataInfo object
  DataInfo<TYPE>* datainfo = new DataInfo<TYPE>();
  datainfo->rows = rows;
  datainfo->cols = cols;
  datainfo->data = data;
  return datainfo;
}
