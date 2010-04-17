#include <algorithm>
#include <iostream>
#include <sstream>
#include <boost/algorithm/string/replace.hpp>
#include "ParseUtils.h"

using namespace std;
using boost::algorithm::replace_all;

// constructor
ParseUtils::ParseUtils(string outfname) {
  string cudafname = outfname + ".cu";
  string cppfname = outfname + ".cpp";
  string headerfname = outfname + ".h";

  // initialize files so that it's blank (with ios::trunc)
  //cudafile.open(cudafname.c_str(), ios::trunc);
  cudafile = new FileUtils(cudafname);
  cppfile = new FileUtils(cppfname);
  headerfile = new FileUtils(headerfname);

  parselibs_included = cudalang_included = false;

  indent = 0; // initially no indent offset 
  fcncount = 0; // no functions yet
} 

// destructor
ParseUtils::~ParseUtils(){
  // delete FileUtils
  delete cudafile;
  delete cppfile;
  delete headerfile;

  // close files
  //cudafile.close(); 
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
 * Consolidates the namespace and beginning of main(). Returns a string which
 * gets put into a vector so we can add more later.
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
 * We're done with main(), so return a string with the `return 0` standard
 * stuff and a close brace.
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
j* returned to be stored in a vector. The vector is eventually written out to
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

  // cpp includes
  string cpp_includes;
  cpp_includes += add_include("<fstream>");
  cpp_includes += add_include("<iostream>");
  cpp_includes += add_include("<string>");
  cpp_includes += add_include("\"" + headerfile->fname() + "\"");
  cppfile->pushInclude(cpp_includes);

  // also throw in "using namespace std" into cpp function declarations, which
  // will always come before the definition but after all the includes
  string namespace_std = prep_str("using namespace std;");
  cppfile->pushFcnDec(namespace_std);
}


/********************************
 * Function: initializeHeader
 * --------------------------
 * Initializes header file with the customary #define syntax
 */
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
}

void ParseUtils::finalizeHeader(){
  int old_indent = indent;
  indent = 0;

  string endheader;
  endheader += prep_str("#endif");

  // adding #endif to the "Function Definition" part of a .h file ensures that
  // it comes last
  headerfile->pushFcnDef(endheader);

  indent = old_indent;
}
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
 * Function: numbered_fcnname
 * --------------------------
 * given a function name, append the value of the current function counter and
 * increment the counter
 */
string ParseUtils::numbered_fcnname(string fcnname){
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
  return on;
}

// TODO DEFUNCT!
/********************************
 * Function: insertVerbatim
 * ------------------------
 * The user can insert code verbatim into their program. For example, to
 * initialize an array the user might write the following code in our
 * language:
 *    STARTCPP
 *      int* data = new data[100];
 *      for (int i=0; i<100; i++){
 *        data[i] = i;
 *      }
 *    ENDCPP
 */
void ParseUtils::insertVerbatim(string object, string code, string type){
  // get the length of each special symbol
  string startsym = "STARTCPP"; int startlen = startsym.length();
  string endsym = "ENDCPP"; int endlen = endsym.length();

  // code length is everything except start and end symbol
  int codelen = code.length() - (startlen + endlen);

  // find instance of start symbol
  int startpos = code.find(startsym);

  // isolate only the relevant code
  code = code.substr(startpos+startlen, codelen);

  // now replace all instances of the object with object_host defined by
  // get_obj_names()
  obj_names objnames = get_obj_names(object);
  string host = objnames.host;
  string dev = objnames.dev;
  string rows = objnames.rows;
  string cols = objnames.cols;
  string datainfo = objnames.datainfo;

  int pos=0;
	while((pos=code.find(object, pos))!=string::npos)
	{
		code.erase(pos, object.length());
		code.insert(pos, host);
		pos+=host.length();
	}

  string code_outstr;
  code_outstr += prep_str(code);
  code_outstr += "\n";

  if (symtab.addEntry(object, VARIABLE, type)){
    // we assume that the user provided the host declaration in the given code
    code_outstr += prep_str(type + "* " + dev + "; // Memory on device (gpu) side");
    code_outstr += prep_str("int " + rows + ", " + cols + ";");
    code_outstr += prep_str("DataInfo<" + type + ">* " + datainfo + ";");
    code_outstr += "\n";
  }
  cudafile->pushMain(code_outstr);
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

  string cuda_outstr;
  if (symtab.addEntry(object, VARIABLE, type)){
    cuda_outstr += prep_str("host_vector<" + type + "> " + host + "; // Memory on host (cpu) side");
    cuda_outstr += prep_str("device_vector<" + type + "> " + dev + "; // Memory on device (gpu) side");
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
  if (symtab.addEntry(object, VARIABLE, type)){
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
  
  string cuda_outstr;
  if (symtab.addEntry(object, VARIABLE, type)){
    cuda_outstr += prep_str("host_vector<" + type + "> " + host + "; // Memory on host (cpu) side");
    cuda_outstr += prep_str("device_vector<" + type + "> " + dev + "; // Memory on device (gpu) side");
    cuda_outstr += prep_str("int " + rows + ", " + cols + ";");
    cuda_outstr += "\n";
  } // TODO: else we need to delete some memory? i.e. host has already been allocated

  cuda_outstr += prep_str("// Set rows, cols");
  cuda_outstr += prep_str(rows + " = " + datarows + ";");
  cuda_outstr += prep_str(cols + " = " + datacols + ";");
  cuda_outstr += "\n";

  cuda_outstr += prep_str(host + ".clear(); // make sure vector is empty");
  cuda_outstr += prep_str("// populate data");
  cuda_outstr += prep_str("for (int row=0; row<" + rows + "; row++){"); indent++;
  cuda_outstr += prep_str("for (int col=0; col<" + cols + "; col++){"); indent++;
  cuda_outstr += prep_str(expr); indent--;
  cuda_outstr += prep_str("}"); indent--;
  cuda_outstr += prep_str("}");
  cuda_outstr += prep_str(dev + " = " + host + "; // copy host data to GPU");
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
  replace_all(expr, "maxcols", decimalcast + objnames.cols);

  replace_all(expr, "value", objnames.host); // + "[row*" + objnames.cols + "+col]");
  replace_all(expr, "=", ".push_back(");
  // TODO: implement rand

  expr += ");";
  return expr;
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
void ParseUtils::readDatafile(string fname, string object, string type){
  // define names of vars in prog
  obj_names objnames = get_obj_names(object);
  string host = objnames.host;
  string dev = objnames.dev;
  string rows = objnames.rows;
  string cols = objnames.cols;
  string datainfo = objnames.datainfo;
  string instream = objnames.instream;
  string garbage = objnames.garbage;

  string fcnname = numbered_fcnname("readDatafile");
  int old_indent = indent;
  indent = 0;

  if (!cudalang_included){
    // make sure cu file has cudalang.h included FIXME: won't always be
    // cudalang
    string cudalang = add_include("\"" + headerfile->fname() + "\"");
    cudafile->pushInclude(cudalang);
    cudalang_included = true;
  }

  if (!parselibs_included){
    // make sure both header file and cu file have included ParseLibs
    string parselibs = add_include("\"ParseLibs.h\"");
    headerfile->pushInclude(parselibs);
    cudafile->pushInclude(parselibs);
    parselibs_included = true;
  }

  // write out function declaration to header file
  string header_outstr;
  header_outstr += prep_str("DataInfo<" + type + ">* " + fcnname + "(char* fname);");
  headerfile->pushFcnDec(header_outstr);

  // write out function definition in cpp file
  string cpp_outstr;
  cpp_outstr += prep_str("DataInfo<" + type + ">* " + fcnname + "(char* fname){"); indent++;
  cpp_outstr += prep_str("ifstream instream;");
  cpp_outstr += prep_str("instream.open(fname);");
  cpp_outstr += "\n";

  cpp_outstr += prep_str("// read in rows and column data");
  cpp_outstr += prep_str("string garbage;");
  cpp_outstr += prep_str("int rows, cols;");
  cpp_outstr += prep_str("instream >> garbage; instream >> rows;");
  cpp_outstr += prep_str("instream >> garbage; instream >> cols;");
  cpp_outstr += "\n";

  cpp_outstr += prep_str("// read in data from file");
  cpp_outstr += prep_str("thrust::host_vector<" + type + "> data(rows*cols);");
  cpp_outstr += prep_str("for (int r=0; r<rows; r++){"); indent++;
  cpp_outstr += prep_str("for (int c=0; c<cols; c++){"); indent++;
  cpp_outstr += prep_str("instream >> data[r*cols+c];"); indent--;
  cpp_outstr += prep_str("}"); indent--;
  cpp_outstr += prep_str("}");
  cpp_outstr += prep_str("instream.close();");
  cpp_outstr += "\n";

  cpp_outstr += prep_str("// package data information in DataInfo object");
  cpp_outstr += prep_str("DataInfo<" + type + ">* datainfo = new DataInfo<" + type + ">();");
  cpp_outstr += prep_str("datainfo->rows = rows;");
  cpp_outstr += prep_str("datainfo->cols = cols;");
  cpp_outstr += prep_str("datainfo->data = data;");
  cpp_outstr += prep_str("return datainfo;"); indent--;
  cpp_outstr += prep_str("}");
  cpp_outstr += prep_str("\n");
  cppfile->pushFcnDef(cpp_outstr);

  indent = old_indent; // restore indent
  
  string cuda_outstr;
  if (symtab.addEntry(object, VARIABLE, type)){
    cuda_outstr += prep_str("host_vector<" + type + "> " + host + "; // Memory on host (cpu) side");
    cuda_outstr += prep_str("device_vector<" + type + "> " + dev + "; // Memory on device (gpu) side");
    cuda_outstr += prep_str("int " + rows + ", " + cols + ";");
    cuda_outstr += prep_str("DataInfo<" + type + ">* " + datainfo + ";");
    cuda_outstr += "\n";
  } // TODO: else we need to delete some memory? i.e. host has already been allocated

  cuda_outstr += prep_str("// Read in data");
  cuda_outstr += prep_str(datainfo + " = " 
      + fcnname + "(\"" + fname + "\");");
  cuda_outstr += prep_str(host + " = " + datainfo + "->data;");
  cuda_outstr += prep_str(rows + " = " + datainfo + "->rows;");
  cuda_outstr += prep_str(cols + " = " + datainfo + "->cols;");
  //outstr += prep_str(host + " = " + fcnname + "(\"" + fname + "\");");
  cuda_outstr += prep_str(dev + " = " + host + "; // copy host data to GPU");
  cuda_outstr += "\n";
  cudafile->pushMain(cuda_outstr);

/*
  outstr += prep_str("// Allocate memory on the host that the gpu can access quickly");
  outstr += prep_str("cudaMallocHost((void**)&" + host + ", " 
      + rows + "*" + cols + "*sizeof(" + type + "));");
  outstr += "\n";
*/

/*
  outstr += prep_str(type + "** " + object + " = new " + type + "*[" + rows + "];");
  outstr += prep_str("for (int r=0; r< " + rows + "; r++){"); indent++;
  outstr += prep_str(object + "[r] = new " + type + "[" + cols + "];"); indent--;
  outstr += prep_str("}");
  outstr += "\n";
*/
}

/********************************
 * Function: writeDatafile
 * -----------------------
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
void ParseUtils::writeDatafile(string fname, string object){
  string outstr;

  // define names of vars in prog
  obj_names objnames = get_obj_names(object);
  string host = objnames.host;
  string rows = objnames.rows;
  string cols = objnames.cols;
  string outstream = objnames.outstream;
  string type = symtab.getType(object);

  string fcnname = numbered_fcnname("writeDatafile");
  int old_indent = indent;
  indent = 0;

  if (!cudalang_included){
    // make sure cu file has cudalang.h included FIXME: won't always be
    // cudalang
    string cudalang = add_include("\"" + headerfile->fname() + "\"");
    cudafile->pushInclude(cudalang);
    cudalang_included = true;
  }

  if (!parselibs_included){
    // make sure both header file and cu file have included ParseLibs
    string parselibs = add_include("\"ParseLibs.h\"");
    headerfile->pushInclude(parselibs);
    cudafile->pushInclude(parselibs);
    parselibs_included = true;
  }
  
  // write out function declaration to header file
  string header_outstr;
  header_outstr += prep_str("void " + fcnname + "(char* fname, " + type + "* data, int rows, int cols);");
  headerfile->pushFcnDec(header_outstr);

  string cpp_outstr;
  cpp_outstr += prep_str("void " + fcnname + "(char* fname, " + type + "* data, int rows, int cols){"); indent++;
  cpp_outstr += prep_str("// output data to file");
  cpp_outstr += prep_str("ofstream outstream;");
  cpp_outstr += prep_str("outstream.open(\"" + fname + "\");");
  cpp_outstr += "\n";

  cpp_outstr += prep_str("// write data out to file");
  cpp_outstr += prep_str("outstream << \"rows: \" << rows << \"\\n\";");
  cpp_outstr += prep_str("outstream << \"cols: \" << cols << \"\\n\";");

  cpp_outstr += prep_str("for (int r=0; r<rows; r++){"); indent++;
  cpp_outstr += prep_str("for (int c=0; c<cols; c++){"); indent++;
  cpp_outstr += prep_str("outstream << data[r*cols+c] << \" \";"); indent--;
  cpp_outstr += prep_str("}"); 
  cpp_outstr += prep_str("outstream << \"\\n\";"); indent--;
  cpp_outstr += prep_str("}");
  cpp_outstr += prep_str("outstream.close();"); indent--;
  cpp_outstr += prep_str("}");
  cpp_outstr += "\n";
  cppfile->pushFcnDef(cpp_outstr);

  indent = old_indent; // restore indent

  string cuda_outstr;
  cuda_outstr += prep_str("// write data out to disk");
  cuda_outstr += prep_str(fcnname + "(\"" + fname + "\", " + host +
      ", " + rows + ", " + cols + ");");
  cuda_outstr += "\n";
  cudafile->pushMain(cuda_outstr);

}

/********************************
 * Function: mapFcn
 * ----------------
 * Generates code for a map across an object
 */
void ParseUtils::mapFcn(string fcnname, string object, string op, string alter){
  int old_indent = indent; // save indent level
  indent = 0; // since we're defining a function, we start at no indent
  fcnname = numbered_fcnname(fcnname);

  // define names of vars in prog
  obj_names objnames = get_obj_names(object);
  string host = objnames.host;
  string dev = objnames.dev;
  string rows = objnames.rows;
  string cols = objnames.cols;
  string type = symtab.getType(object);

  // function declaration
  // example:
  //    int** map(int** data, int rows, int cols);
  string declaration; 
  declaration += prep_str("__global__ void " + fcnname + "(" 
    + type + "* data, int cols);");
  cudafile->pushFcnDec(declaration);

  // function definition
  string definition;
  definition += prep_str("__global__ void " + fcnname + "(" 
    + type + "* data, int cols){"); indent++;

  definition += prep_str("int row = threadIdx.x;");
  definition += prep_str("int col = threadIdx.y;");
  definition += "\n";

  definition += prep_str("// operate on data");
  definition += prep_str("data[row*cols + col] " + op + "= " + alter + ";"); indent--;
  definition += prep_str("}");
  definition += "\n";
  cudafile->pushFcnDef(definition);
  
  indent = old_indent; // restore indent level

  // function call (in main)
  string call;
  call += prep_str("// Allocate memory on the device that is of the same size as our host array");
  call += prep_str("cudaMalloc((void**)&" + dev + ", " 
      + rows + "*" + cols + "*sizeof(" + type + "));");
  call += "\n";

  call += prep_str("// Copy host to device");
  call += prep_str(
      cudamemcpy_str(
        dev, 
        host, 
        rows + "*" + cols + "*sizeof(" + type + ")",
        "cudaMemcpyHostToDevice"));
  call += "\n";

  call += prep_str("// Run the '" + fcnname + "' kernel on the device");
  call += prep_str("// 1 tells it to have one single block");
  call += prep_str("// dim3(row,col) tells it to have row by col threads in that block");
  call += prep_str("// Give kernel the 'dev' array and the number of cols");
  call += prep_str(fcnname + "<<<1, dim3(" + 
      rows + "," + cols + ")>>>(" + dev + "," + cols + ");");
  call += "\n";


  call += prep_str("// Once the kernel has run, copy back dev to host");
  call += prep_str(
      cudamemcpy_str(
        host, 
        dev, 
        rows + "*" + cols + "*sizeof(" + type + ")",
        "cudaMemcpyDeviceToHost"));
  call += "\n";
    
  cudafile->pushMain(call);
}

/********************************
 * Function: cudamemcpy_str
 * ------------------------
 * Since the cudaMemcpy calls are really gross, we run the four parameters
 * through this string processing function to make the process a bit cleaner.
 *
 * Inputs:
 *    to: variable to copy memory to
 *    from: variable to copy memory from
 *    size: size of data being copied
 *    directive: a CUDA directive 
 *        (i.e. "cudaMemcpyHostToDevice")
 *
 * Returns:
 *    fully realized cudaMemcpy command
 */
string ParseUtils::cudamemcpy_str(
    string to, string from, string size, string directive){
  return "cudaMemcpy(" 
    + to + ", " 
    + from + ", " 
    + size + ", " 
    + directive + ");";
}


