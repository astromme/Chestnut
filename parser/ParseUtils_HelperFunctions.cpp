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
 * Function: writeAllFiles
 * -----------------------
 * Writes contents of files to disk using FileUtils
 */
void ParseUtils::writeAllFiles(){
  cudafile->writeAll();
  cppfile->writeAll();
  headerfile->writeAll();
}

/********************************
 * Function: initializeMain
 * ------------------------
 * Consolidates the namespace and beginning of main(). 
 */
void ParseUtils::initializeMain(){
  string startmain;
  startmain += "using namespace thrust;\n\n";
  startmain += "int main() {\n";

  cudafile->pushMain(startmain);
  indent++; // add indentation level
}


/********************************
 * Function: finalizeMain
 * ----------------------
 * We're done with main(), so push "return 0" standard and a close brace.
 */
void ParseUtils::finalizeMain(){
  string endmain;
  endmain += "  return 0;\n";
  endmain += "}\n"; // close main()

  cudafile->pushMain(endmain);
}

/********************************
 * Function: initializeIncludes
 * ----------------------------
 * Consolidates some baseline #includes. These are stored as strings and
 * returned to be stored in a vector. The vector is eventually written out to
 * the file, but in case we want to add more #includes later on, we can.
 */
void ParseUtils::initializeIncludes(){
  // add #include infos
  string includes;
  /*includes += add_include("string");
  includes += add_include("fstream");
  includes += add_include("iostream");
  includes += "\n";*/

  // cudafile includes
  string cuda_includes;
  cuda_includes += prep_str("// CUDA includes");
  cuda_includes += add_include("<cuda.h>");
  cuda_includes += add_include("<cuda_runtime_api.h>");
  cuda_includes += prep_str("// Thrust includes");
  cuda_includes += add_include("<thrust/host_vector.h>");
  cuda_includes += add_include("<thrust/device_vector.h>");
  cuda_includes += add_include("<thrust/iterator/constant_iterator.h>");
  cudafile->pushInclude(cuda_includes);

  // cpp includes  TODO remove
  /*string cpp_includes;
  cpp_includes += add_include("<fstream>");
  cpp_includes += add_include("<iostream>");
  cpp_includes += add_include("<string>");
  cpp_includes += add_include("\"" + headerfile->fname() + "\"");
  cppfile->pushInclude(cpp_includes);*/

  // also throw in "using namespace std" into cpp function declarations, which
  // will always come before the definition but after all the includes
  /*string namespace_std = prep_str("using namespace std;");
  cppfile->pushFcnDec(namespace_std);*/
}

// TODO remove
/********************************
 * Function: initializeHeader
 * --------------------------
 * Initializes header file with the customary #define syntax
 *
void ParseUtils::initializeHeader(){
  string headername = headerfile->fname();

  // convert to uppercase
  transform(headername.begin(), headername.end(), headername.begin(), ::toupper);

  // replace .H with _H
  headername.replace(headername.length()-2,1,"_");

  int old_indent = indent;
  indent = 0;

  string define;
  define += prep_str("#ifndef _" + headername);
  define += prep_str("#define _" + headername);
  define += "\n";
  headerfile->pushInclude(define);

  string includes;
  headerfile->pushInclude(includes);

  indent = old_indent;
}*/

/* TODO remove
void ParseUtils::finalizeHeader(){
  int old_indent = indent;
  indent = 0;

  string endheader;
  endheader += prep_str("#endif");

  // adding #endif to the "Function Definition" part of a .h file ensures that
  // it comes last
  headerfile->pushFcnDef(endheader);

  indent = old_indent;
}*/

/********************************
 * Function: add_include
 * ---------------------
 * given a header file, wraps it in the customary #include<foo> syntax and
 * returns it.
 */
string ParseUtils::add_include(string header){
  return "#include " + header + "\n";
}

/********************************
 * Function: prep_str
 * ------------------
 * prepares a string for file insertion by appending a newline and indenting
 * the proper amount, based on the global indent offset. Returns the indented,
 * "newlined" string
 */
string ParseUtils::prep_str(string str) { 
  string indentstr = "";
  for (int i=0; i<indent; i++){
    indentstr += "  ";
  }
  return indentstr + str + "\n";
}

/********************************
 * Function: numberedId
 * --------------------
 * given a id name, append the value of the current function counter and
 * increment the counter
 */
string ParseUtils::numberedId(string fcnname){
  stringstream ss; ss << fcncount;
  string new_fcnname = fcnname + ss.str();
  fcncount++;

  return new_fcnname;
}

/********************************
 * Function: get_obj_names
 * -----------------------
 * We want to have a consistent naming scheme for our variables, so we
 * centralize the process by passing it through this function. Since every
 * object in a given scope must be uniquely named, we use the variable name
 * given in the source language as the base for our naming scheme.
 *
 * Input:
 *    obj: name of object
 *
 * Returns:
 *    struct containing all of the derived names that we might want    
 */
obj_names ParseUtils::get_obj_names(string obj){
  obj_names on;
  on.host = obj + "_host";
  on.dev = obj + "_dev";
  on.rows = obj + "_rows";
  on.cols = obj + "_cols";
  on.datainfo = obj + "_datainfo";
  on.instream = obj + "_instream";
  on.outstream = obj + "_outstream";
  on.garbage = obj + "_garbage";
  on.timerstart = obj + "_timerstart";
  on.timerstop = obj + "_timerstop";
  return on;
}

/********************************
 * Function: copyDevData
 * ---------------------
 * Returns code that copies data from one thrust::device_vector to another. We
 * might want to do this in the case of a map() where the user does not want
 * to modify the source vector. Code would look like
 *    data_mapped = map(+, 1, data_source)
 *
 * Inputs:
 *    source: data that will be copied
 *    destination: destination of copied data
 *
 * Returns:
 *    string containing the code necessary to copy destination to source
 */
string ParseUtils::copyDevData(string source, string destination){
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

  string copy_outstr;
  if (dest_type != src_type){
    // throw exception because dest_type needs to be same as src_type for
    // copy to succeed
    char error_msg[100+destination.length()+source.length()];
    sprintf(error_msg, "in makeMap: destination (%s) and source (%s) are different types", 
        destination.c_str(), source.c_str());
    throw runtime_error(error_msg);
  }

  copy_outstr += prep_str(dest_dev + " = " + src_dev + "; // copy destination to source");

  // also need to make sure rows/cols are set
  copy_outstr += prep_str(dest_rows + " = " + src_rows + ";");
  copy_outstr += prep_str(dest_cols + " = " + src_cols + ";");
  copy_outstr += "\n";

  return copy_outstr;
}

/********************************
 * Function: getThrustOp
 * ---------------------
 * Given one of the following operations
 *    "+" "-" "*" "/" "%"
 * we need to turn it into a thrust-ready function, for example:
 *    "+" becomes "plus<type>()"
 *
 * Inputs:
 *    op: operation as described above
 *    type: type that operation will be acting on
 */
string ParseUtils::getThrustOp(string op, string type){
  string thrustop = "thrust::";
  if (op == "+") thrustop += "plus";
  else if (op == "-") thrustop += "minus";
  else if (op == "*") thrustop += "multiplies";
  else if (op == "/") thrustop += "divides";
  else if (op == "%") thrustop += "modulus";
  else if (op == "<") thrustop += "less";
  else if (op == ">") thrustop += "greater";
  else if (op == "<=") thrustop += "less_equal";
  else if (op == ">=") thrustop += "greater_equal";
  else {
    char error_msg[100+op.length()];
    sprintf(error_msg, "in getThrustOp: unrecognized operator '%s'",
      op.c_str());
    throw runtime_error(error_msg);
  }
  thrustop += "<" + type + ">()";
  return thrustop;
}