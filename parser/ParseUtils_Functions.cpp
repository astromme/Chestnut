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
 * Function: makeMap
 * -----------------
 * Generates code for a map across an object
 */
void ParseUtils::makeMap(string source, string destination, string op, string modify){
  
  /* if destination and modify are the same variable, we want to switch the order 
   * i.e. 
   *     mod = map(+,mod,data);
   * becomes
   *     mod = map(+,data,mod);
   */
  if (destination == modify){
    string tmpModify = modify;
    modify = source;
    source = tmpModify;
  }
  
  // define names of vars in prog
  obj_names src_objnames = get_obj_names(source);
  string src_dev = src_objnames.dev;
  string src_rows = src_objnames.rows;
  string src_cols = src_objnames.cols;
  string src_type = symtab.getType(source);

  obj_names dest_objnames = get_obj_names(destination);
  string dest_dev = dest_objnames.dev;
  string dest_rows = dest_objnames.rows;
  string dest_cols = dest_objnames.cols;
  string dest_type = symtab.getType(destination);

  string thrustop = getThrustOp(op, dest_type);

  string cuda_outstr;
 
  cuda_outstr += prep_str("/* Begin Map Function */");

  // if destination is not source -- if we're not modifying data in place but
  // instead want to leave the original unchanged -- we need to allocate space onto 
  // destination and then operate on that
  if (destination != source){
    if (dest_type != src_type){
      // throw exception because dest_type needs to be same as src_type for
      // copy to succeed
      char error_msg[100+destination.length()+source.length()];
      sprintf(error_msg, "in makeMap: destination (%s) and source (%s) are different types", 
          destination.c_str(), source.c_str());
      throw runtime_error(error_msg);
    }
    
    // set destination size
    cuda_outstr += prep_str(dest_dev + " = device_vector<" + src_type + ">(" + 
      src_dev + ".size());");
    
    // need to make sure rows/cols are set
    cuda_outstr += prep_str(dest_rows + " = " + src_rows + ";");
    cuda_outstr += prep_str(dest_cols + " = " + src_cols + ";");
    cuda_outstr += "\n";
  }

  // if 'modify' is not in the symtab, then we need to 
  // create a constant iterator that returns the given value
  string mapmodify = get_obj_names(modify).dev + ".begin()";
  if (!symtab.isInSymtab(modify)){
    mapmodify = "thrust::make_constant_iterator(" + modify + ")";
  }

  // call actual map function (using thrust::transform)
  cuda_outstr += prep_str("thrust::transform(" +
      src_dev + ".begin(), " + src_dev + ".end(), " + // source is
      mapmodify + ", " + // combined with mapmodify and
      dest_dev + ".begin(), " + // written to destination.
      thrustop + ");"); // source and mapmodify are compsed with thrustop
  cuda_outstr += prep_str("/* End Map Function */"); 
  cuda_outstr += "\n";

  cudafile->pushMain(cuda_outstr);

}

/********************************
 * Function: makeReduce
 * --------------------
 * Generates code for a reduce of an object
 */
void ParseUtils::makeReduce(string source, string destination, string op){
  // define names of vars in prog
  obj_names src_objnames = get_obj_names(source);
  string src_dev = src_objnames.dev;
  string src_type = symtab.getType(source);

  obj_names dest_objnames = get_obj_names(destination);
  string dest_host = dest_objnames.host;
  string dest_type = symtab.getType(destination);

  string thrustop = getThrustOp(op, dest_type);

  string cuda_include = add_include("<thrust/reduce.h>");
  cudafile->pushInclude(cuda_include);

  string cuda_outstr;
  cuda_outstr += prep_str("/* Begin Reduce Function */");

  // throw exception because dest_type needs to be same as src_type for
  // copy to succeed
  if (dest_type != src_type){
    char error_msg[100+destination.length()+source.length()];
    sprintf(error_msg, "in makeMap: destination (%s) and source (%s) are different types", 
        destination.c_str(), source.c_str());
    throw runtime_error(error_msg);
  }

  // call actual reduce function
  cuda_outstr += prep_str(dest_host + " = thrust::reduce(" +
      src_dev + ".begin(), " + src_dev + ".end(), " + // source is
      "(" + src_type + ") 0, " + // initial value -- almost always want this to be zero
      thrustop + ");"); // source and mapmodify are compsed with thrustop
  cuda_outstr += prep_str("/* End Reduce Function */");
  cuda_outstr += "\n";

  cudafile->pushMain(cuda_outstr);
}

/********************************
 * Function: makeSort
 * ------------------
 * Generates code for sorting data
 */
void ParseUtils::makeSort(string source, string destination, string comparator){
  obj_names src_objnames = get_obj_names(source);
  string src_dev = src_objnames.dev;
  string src_type = symtab.getType(source);

  obj_names dest_objnames = get_obj_names(destination);
  string dest_dev = dest_objnames.dev;
  string dest_type = symtab.getType(destination);
  
  comparator = getThrustOp(comparator, src_type);
  
  string cuda_include = add_include("<thrust/sort.h>");
  cudafile->pushInclude(cuda_include);

  string cuda_outstr;
  cuda_outstr += prep_str("/* Begin Sort Function */");

  if (source != destination){
    cuda_outstr += copyDevData(source, destination);
  }

  cuda_outstr += prep_str("thrust::sort(" +
      dest_dev + ".begin(), " +
      dest_dev + ".end(), " +
      comparator + "); // sort data");

  cuda_outstr += prep_str("/* End Sort Function */");
  cuda_outstr += "\n";

  cudafile->pushMain(cuda_outstr);
}

