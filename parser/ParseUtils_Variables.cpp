#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <boost/algorithm/string/replace.hpp>
#include <boost/regex.hpp>
#include "ParseUtils.h"

using namespace std;
using boost::algorithm::replace_all;

/********************************
 * Function: makeVector
 * --------------------
 * Assocated with the syntax in Chestnut
 *    <type> <name> vector
 *
 * These are variables that will be used in the program execution but are not
 * inititalized from the outset. For example, if you map across some data A
 * and you want to send it to B, we would declare B here
 *
 * Inputs:
 *    object: name of vector
 *    type: type of vector
 */ 
void ParseUtils::makeVector(string object, string type){
  // define names of vars in prog
  obj_names objnames = get_obj_names(object);
  string host = objnames.host;
  string dev = objnames.dev;
  string rows = objnames.rows;
  string cols = objnames.cols;

  string cuda_outstr;
  if (symtab.addEntry(object, VARIABLE_VECTOR, type)){
    cuda_outstr += prep_str("host_vector<" + type + "> " + host + "; // Memory on host (cpu) side");
    cuda_outstr += prep_str("device_vector<" + type + "> " + dev + "; // Memory on device (gpu) side");
    cuda_outstr += prep_str("int " + rows + ", " + cols + ";");
    cuda_outstr += "\n";
  } // TODO: else we need to delete some memory? i.e. host has already been allocated
  cudafile->pushMain(cuda_outstr);
}

/********************************
 * Function: makeScalar
 * --------------------
 * Assocated with the syntax in Chestnut
 *    <type> <name> scalar
 *
 * These are variables that will be used in the program execution but are not
 * inititalized from the outset. For example, if you map across some data A
 * and you want to send it to B, we would declare B here
 *
 * Inputs:
 *    object: name of scalar
 *    type: type of scalar
 */ 
void ParseUtils::makeScalar(string object, string type){
  // define names of vars in prog
  obj_names objnames = get_obj_names(object);
  string host = objnames.host;

  string cuda_outstr;
  if (symtab.addEntry(object, VARIABLE_SCALAR, type)){
    cuda_outstr += prep_str(type + " " + host + "; // scalar on host (cpu) side");
    cuda_outstr += "\n";
  } // TODO: else we need to delete some memory? i.e. host has already been allocated
  cudafile->pushMain(cuda_outstr);
}