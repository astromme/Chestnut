#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <boost/algorithm/string/replace.hpp>
#include <boost/regex.hpp>
#include "ParseUtils.h"

using namespace std;
using boost::algorithm::replace_all;

// constructor
ParseUtils::ParseUtils(string outfname) {
  string cudafname = outfname + ".cu";
  string cppfname = outfname + ".cpp";
  string headerfname = outfname + ".h";

  // initialize files so that it's blank (with ios::trunc)
  cudafile = new FileUtils(cudafname);
  cppfile = new FileUtils(cppfname);
  headerfile = new FileUtils(headerfname);

  parselibs_included = false;
  // cudalang_included = false; // TODO Don't need cudalang?
  rand_included = false;

  indent = 0; // initially no indent offset 
  fcncount = 0; // no functions yet
} 

// destructor
ParseUtils::~ParseUtils(){
  // delete FileUtils
  delete cudafile;
  delete cppfile;
  delete headerfile;
}

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
 * Function: makeVector
 * --------------------
 * Assocated with the syntax in Chestnut
 *    <type> <name> vector
 *
 * These are variables that will be used in the program execution but are not
 * inititalized from the outset. For example, if you map across some data A
 * and you want to send it to B, we would declare B here
 *
 *
 * TODO: Inputs/Returns
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
 *
 * TODO: Inputs/Returns
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

/********************************
 * Function: makeForeach
 * ---------------------
 * Associated with the syntax in Chestnut
 *      <type> <name> <rows> <cols> foreach ( <expr> );
 *
 * Generates code to fill data with given expression.
 *
 * FIXME: more specific
 * Expression can consist of a bunch of stuff like rows, cols, maxrows, etc
 * that we have to deal with
 *
 * Inputs:
 *    object: name of variable to be initialized
 *    type: type of var (int, float, etc)
 *    datarows: number of rows to initialize
 *    datacols: number of columns to initialize
 *    expr: expression used to populate variable
 */
void ParseUtils::makeForeach(string object, string type, string datarows, string datacols, string expr){
  // define names of vars in prog
  obj_names objnames = get_obj_names(object);
  string host = objnames.host;
  string dev = objnames.dev;
  string rows = objnames.rows;
  string cols = objnames.cols;
  expr = processForeachExpr(expr, objnames);

  /* We want to check if the keywords "row" or "col" are in the given
   * expression. If they are, we need to use the expanded for-loop syntax to
   * populate the data on the CPU and then copy over to GPU. This is slow.
   *
   * If those keywords are not in the expression, then we know all of the
   * variable values without the need of a for-loop. We can thus optimize this
   * by using thrust::fill.
   */
  
  // right hand side of expression
  string expr_rhs = expr.substr(expr.find("=")+1,expr.length()); 
  boost::regex row_or_col("\\<row\\>|\\<col\\>");

  bool optimize_fill = true;
	if (boost::regex_search (expr_rhs, row_or_col)){
    optimize_fill = false;
  }

  // check if rand exists, and, if so, whether we have seeded the random
  // number generator
  boost::regex rand("\\<rand\\>");
  if (boost::regex_search (expr_rhs, rand)){
    optimize_fill = false;

    if (!rand_included){
      string randincl;
      randincl += add_include("<cstdlib>");
      randincl += add_include("<ctime>");
      cudafile->pushInclude(randincl);

      string randstr = prep_str("srand( time(NULL) ); // seed random number generator");
      randstr += "\n";
      cudafile->pushMain(randstr);

      rand_included = true;
    }
  }
  
  string cuda_outstr;
  if (symtab.addEntry(object, VARIABLE_VECTOR, type)){
    cuda_outstr += prep_str("int " + rows + " = " + datarows + ";");
    cuda_outstr += prep_str("int " + cols + " = " + datacols + ";"); 

    // if we decided to optimize, we don't want to waste time allocating space
    // on the host_vector
    string host_declaration = "host_vector<" + type + "> " + host;
    if (!optimize_fill){
      host_declaration += "(" + rows + "*" + cols + ")";
    }
    host_declaration += "; // Memory on host (cpu) side";
    cuda_outstr += prep_str(host_declaration);

    // need to allocate space on device regardless of optimization
    cuda_outstr += prep_str("device_vector<" + type + "> " + dev + 
        "(" + rows + "*" + cols + "); // Memory on device (gpu) side");
    cuda_outstr += "\n";
  } // TODO: else we need to delete some memory? i.e. host has already been allocated

  cuda_outstr += prep_str("// populate data");
  // if we can optimize using fill, do so
  if (optimize_fill){
    cuda_outstr += prep_str("thrust::fill(" + dev + ".begin(), " +
        dev + ".end(), " + expr_rhs + ");");
  } else {
    cuda_outstr += prep_str("for (int row=0; row<" + rows + "; row++){"); indent++;
    cuda_outstr += prep_str("for (int col=0; col<" + cols + "; col++){"); indent++;
    cuda_outstr += prep_str(expr + ";"); indent--;
    cuda_outstr += prep_str("}"); indent--;
    cuda_outstr += prep_str("}");
    cuda_outstr += prep_str(dev + " = " + host + "; // copy host data to GPU");
  }
  cuda_outstr += "\n";
  cudafile->pushMain(cuda_outstr);
}

/********************************
 * Function: processForeachExpr
 * ----------------------------
 * a `foreach` expression can consist of more than just simple arithmetic
 * expressions. We have a set of reserved words which can be used at a high
 * level to indicate lower-level changes. They are as follows.
 *
 * ==> maxrows: total number of rows in data block
 * ==> maxcols: total number of columns in data block
 * ==> row: current row index
 * ==> col: current column index
 * ==> value: current index of array, 
 *    i.e. arr[row][col], represented as arr[row*maxcols+col]
 * ==> rand: a random real number between 0 and 1
 *
 * Thus, as we see these reserved words, we must replace them in the code with
 * the correct names, as given by the object name with which the foreach
 * statement is associated. Also, row and col will be replaced by 'r' and 'c',
 * respectively, which is object-name independent.
 */
string ParseUtils::processForeachExpr(string expr, const obj_names &objnames){
  string decimalcast = "(float)";
  
  // since the right hand side may have something like "row/maxrows" we want
  // that to be a float, not an int, so let's cast variables as floats
  // TODO: make sure this doesn't break anything
  replace_all(expr, "row", decimalcast + "row");
  replace_all(expr, "max" + decimalcast + "rows", decimalcast + objnames.rows);

  replace_all(expr, "col", decimalcast + "col");
  replace_all(expr, "max" + decimalcast + "cols", decimalcast + objnames.cols);

  replace_all(expr, "rand", decimalcast + "rand()");

  replace_all(expr, "value", objnames.host + "[row*" + objnames.cols + "+col]");
  replace_all(expr, "=", " = ");
  //replace_all(expr, "=", ".push_back(");
  // TODO: implement rand

  return expr;
}

/********************************
 * Function: makeTimer
 * -------------------
 * Associated with syntax in Chestnut
 *      timer <name>
 *
 * Generates code to declare timers
 *
 * Inputs:
 *    timer: name of timer
 */
void ParseUtils::makeTimer(string timer)
{
  if (symtab.addEntry(timer, VARIABLE_TIMER, "")){
    obj_names objnames = get_obj_names(timer);
    string start = objnames.timerstart;
    string stop = objnames.timerstop;
  
    // need to include sys/time.h and ParseLibs
    string timerinclude;
    timerinclude += add_include("<sys/time.h>");
    timerinclude += add_include("\"ParseLibs.h\"");
    cudafile->pushInclude(timerinclude);
  
    string cuda_outstr;
    cuda_outstr += prep_str("struct timeval " +
      start + ", " + stop + "; // instantiate timer");
    cuda_outstr += "\n";
    cudafile->pushMain(cuda_outstr);
  }
}

/********************************
 * Function: makeTimerStart
 * ------------------------
 * Associated with syntax in Chestnut
 *      <name> start
 *
 * Generates code to start a timer
 *
 * Inputs:
 *    timer: name of timer
 */
void ParseUtils::makeTimerStart(string timer)
{
  obj_names objnames = get_obj_names(timer);
  string start = objnames.timerstart;
 
  string cuda_outstr;
  cuda_outstr += prep_str("gettimeofday(&" + start + ", 0); // start timer");
  cuda_outstr += "\n";
  cudafile->pushMain(cuda_outstr);
}

/********************************
 * Function: makeTimerStart
 * ------------------------
 * Associated with syntax in Chestnut
 *      <name> stop
 *
 * Generates code to stop a timer
 *
 * Inputs:
 *    timer: name of timer
 */
void ParseUtils::makeTimerStop(string timer)
{
  obj_names objnames = get_obj_names(timer);
  string stop = objnames.timerstop;
 
  string cuda_outstr;
  cuda_outstr += prep_str("gettimeofday(&" + stop + ", 0); // start timer");
  cuda_outstr += "\n";
  cudafile->pushMain(cuda_outstr);
}


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
  string instream = objnames.instream;
  string garbage = objnames.garbage;

  int old_indent = indent;
  indent = 0;
/* TODO remove
  if (!cudalang_included){
    // make sure cu file has cudalang.h included FIXME: won't always be
    // cudalang
    string cudalang = add_include("\"" + headerfile->fname() + "\"");
    cudafile->pushInclude(cudalang);
    cudalang_included = true;
  }
*/

  if (!parselibs_included){
    // make sure both header file and cu file have included ParseLibs
    string parselibs = add_include("\"ParseLibs.h\"");
    // headerfile->pushInclude(parselibs); // TODO remove
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
  } // TODO: else we need to delete some memory? i.e. host has already been allocated

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
 * Associated with the syntax in our language (TODO: name our language)
 *      write(<data>, "<outfile>");
 *
 * Generates code to write data to a file.
 *
 * Inputs:
 *    fname: filename containing data we want to write to
 *    object: object in which data is currently stored
 *    type: datatype of object
 *
 * Returns:
 *    C++ code to implement this data write
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
 * Generates code to print out an object (in one dimension).
 * We don't need to explicitly copy device data onto the CPU, so this function
 * might be a bit faster to run than makePrintData2D
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
void ParseUtils::makePrintData1D(string object){
  // define names of vars in prog
  obj_names objnames = get_obj_names(object);
  string dev = objnames.dev;
  string type = symtab.getType(object);
  
  string cuda_outstr;
  cuda_outstr += prep_str("thrust::copy(" + 
    dev + ".begin(), " + dev + ".end(), " +
    "std::ostream_iterator<" + type + ">(std::cout, \"\\n\"));");
  cuda_outstr += prep_str("std::cout << \"\\n\";");

  cudafile->pushMain(cuda_outstr);
}

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
  // instead want to leave the original unchanged -- we need to copy source
  // into destination and then operate on that
  if (destination != source){
    cuda_outstr += copyDevData(source, destination);

  }

  // if 'modify' is not in the symtab, then we need to create a device_vector
  // full of the given value
  string mapmodify = get_obj_names(modify).dev;
  if (!symtab.isInSymtab(modify)){
    mapmodify = numberedId("mapmodify");
    cuda_outstr += prep_str("// Fill mapmodify with " + modify + "'s");

    cuda_outstr += prep_str("device_vector<" + dest_type + "> " + 
        mapmodify + "(" + dest_dev + ".size());");

    cuda_outstr += prep_str("fill(" + mapmodify + ".begin(), " + 
        mapmodify + ".end(), " + modify + ");");

    cuda_outstr += "\n";
  }

  // call actual map function (using thrust::transform)
  cuda_outstr += prep_str("thrust::transform(" +
      src_dev + ".begin(), " + src_dev + ".end(), " + // source is
      mapmodify + ".begin(), " + // combined with mapmodify and
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

/********************************
 * Function: copyDevData
 * ---------------------
 * Returns code that copies data from one thrust::device_vector to another. We
 * might want to do this in the case of a map() where the user does not want
 * to matify the source vector. Code would look like
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
