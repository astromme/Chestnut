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
 * Function: readDatafile
 * ----------------------
 * Associated with the syntax in Chestnut
 *      <type> <name> read("<infile>");
 *
 * Generates code to read in data from a file.
 *
 * Inputs:
 *    fname: filename containing data we want to read
 *    object: object in which we will store read data
 *    type: type of data we will read in -- thus, datatype of object
 *
 * Returns:
 *    C++ code to implement this data read
 */
// TODO: throw error if file can't be opened
void ParseUtils::makeReadDatafile(string fname, string object, string type){
  // define names of vars in prog
  obj_names objnames = get_obj_names(object);
  string host = objnames.host;
  string dev = objnames.dev;
  string rows = objnames.rows;
  string cols = objnames.cols;
  string datainfo = objnames.datainfo;
  string garbage = objnames.garbage;

  int old_indent = indent;
  indent = 0;

  if (!parselibs_included){
    // make sure both header file and cu file have included ParseLibs
    string parselibs = add_include("\"ParseLibs.h\"");
    cudafile->pushInclude(parselibs);
    parselibs_included = true;
  }

  indent = old_indent; // restore indent
  
  string cuda_outstr;
  if (symtab.addEntry(object, VARIABLE_VECTOR, type)){
    cuda_outstr += prep_str("host_vector<" + type + "> " + host + "; // Memory on host (cpu) side");
    cuda_outstr += prep_str("device_vector<" + type + "> " + dev + "; // Memory on device (gpu) side");
    cuda_outstr += prep_str("int " + rows + ", " + cols + ";");
    cuda_outstr += prep_str("DataInfo<" + type + ">* " + datainfo + ";");
    cuda_outstr += "\n";
  }

  cuda_outstr += prep_str("// Read in data (using ParseLibs)");
  cuda_outstr += prep_str(datainfo + " = readDatafile<" + type +">(\"" + fname + "\");");
  cuda_outstr += prep_str(host + " = " + datainfo + "->data;");
  cuda_outstr += prep_str(rows + " = " + datainfo + "->rows;");
  cuda_outstr += prep_str(cols + " = " + datainfo + "->cols;");
  cuda_outstr += prep_str(dev + " = " + host + "; // copy host data to GPU");
  cuda_outstr += "\n";
  cudafile->pushMain(cuda_outstr);
}

/********************************
 * Function: makewriteDatafile
 * ---------------------------
 * Associated with the syntax in Chestnut
 *      write(<data>, "<outfile>");
 *
 * Generates code to write data to a file.
 *
 * Inputs:
 *    fname: filename containing data we want to write to
 *    object: object where data is stored
 */
void ParseUtils::makeWriteDatafile(string fname, string object){
  string outstr;

  // define names of vars in prog
  obj_names objnames = get_obj_names(object);
  string host = objnames.host;
  string dev = objnames.dev;
  string rows = objnames.rows;
  string cols = objnames.cols;
  string outstream = objnames.outstream;
  string type = symtab.getType(object);

  string fcnname = numberedId("writeDatafile");
  int old_indent = indent;
  indent = 0;

  if (!parselibs_included){
    // make sure both header file and cu file have included ParseLibs
    string parselibs = add_include("\"ParseLibs.h\"");
    headerfile->pushInclude(parselibs);
    cudafile->pushInclude(parselibs);
    parselibs_included = true;
  }
  
  indent = old_indent; // restore indent

  string cuda_outstr;
  cuda_outstr += prep_str("// copy data from gpu to cpu");
  cuda_outstr += prep_str(host + " = " + dev + ";");
  cuda_outstr += prep_str("// write data out to disk (using ParseLibs)");
  cuda_outstr += prep_str("writeDatafile(\"" + fname + "\", " +
      host + ", " + rows + ", " + cols + ");");
  cuda_outstr += "\n";
  cudafile->pushMain(cuda_outstr);

}

/********************************
 * Function: makePrintData
 * -----------------------
 * Generates code to print out an object in two dimensions
 * Unlike makePrintData, we need to copy data over to the CPU from the GPU,
 * but we then gain the ability to structure output in two dimensions
 *
 * For example can print out
 *    1 2 3
 *    4 5 6
 *    7 8 9
 * instead of
 *    1 2 3 4 5 6 7 8 9
 *
 * Input:
 *    object: variable to be printed
 */
void ParseUtils::makePrintData(string object){
  // define names of vars in prog
  obj_names objnames = get_obj_names(object);
  string host = objnames.host;
  string dev = objnames.dev;
  string rows = objnames.rows;
  string cols = objnames.cols;
  string timerstart = objnames.timerstart;
  string timerstop = objnames.timerstop;
  
  string type = symtab.getType(object);
  int category = symtab.getCategory(object);

  string cuda_outstr;
  cuda_outstr += prep_str("// print out data");
  switch (category){
    case VARIABLE_VECTOR:
      cuda_outstr += prep_str("// copy data to cpu then print it");
      cuda_outstr += prep_str(host + " = " + dev + ";");
      cuda_outstr += prep_str("for (int r=0; r<" + rows + "; r++){"); indent++;
      cuda_outstr += prep_str("for (int c=0; c<" + cols + "; c++){"); indent++;
      cuda_outstr += prep_str("std::cout << " + host + "[r*" + cols + "+c] << \" \";"); indent--;
      cuda_outstr += prep_str("}"); 
      cuda_outstr += prep_str("std::cout << \"\\n\";"); indent--;
      cuda_outstr += prep_str("}");
      break;
    case VARIABLE_SCALAR:
      cuda_outstr += prep_str("std::cout << " + host + " << std::endl;");
      break;
    case VARIABLE_TIMER:
      cuda_outstr += prep_str("std::cout << \"timer " + object + ": \" << totalTime(&" + 
         timerstart + ", &" + timerstop + ") << \" seconds\" << std::endl;");
      break;
  }
  cuda_outstr += prep_str("std::cout << \"\\n\";");

  cuda_outstr += "\n";
  cudafile->pushMain(cuda_outstr);
}

/********************************
 * Function: makePrintData1D
 * -------------------------
 * Generates code to print out an object (in one dimension).
 * We don't need to explicitly copy device data onto the CPU, so this function
 * might be a bit faster to run than makePrintData2D
 *
 * Input:
 *    object: variable to be printed
 */
void ParseUtils::makePrintData1D(string object){
  // define names of vars in prog
  obj_names objnames = get_obj_names(object);
  string dev = objnames.dev;
  string type = symtab.getType(object);
  
  string cuda_outstr;
  cuda_outstr += prep_str("thrust::copy(" + 
    dev + ".begin(), " + dev + ".end(), " +
    "std::ostream_iterator<" + type + ">(std::cout, \" \"));");
  cuda_outstr += prep_str("std::cout << \"\\n\";");

  cudafile->pushMain(cuda_outstr);
}
