#include <algorithm>
#include <iostream>
#include <sstream>
#include "ParseUtils.h"

using namespace std;

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
  // use standard namespace (?)
  startmain += "using namespace std;\n\n";
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

/********************************
 * Function: readDatafile
 * ----------------------
 * Associated with the syntax in our language (TODO: name our language)
 *      <data> = read("<infile>");
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
  string outstr;

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
  cpp_outstr += prep_str(type + "* data = new " + type + "[rows*cols];");
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

  indent = old_indent; // restore indent
  outstr += prep_str(type + "* " + host + "; // Memory on host (cpu) side");
  outstr += prep_str(type + "* " + dev + "; // Memory on device (gpu) side");
  outstr += prep_str("int " + rows + ", " + cols + ";");
  outstr += "\n";

  outstr += prep_str("// Read in data");
  outstr += prep_str("DataInfo<" + type + ">* " + datainfo + " = " 
      + fcnname + "(\"" + fname + "\");");
  outstr += prep_str(host + " = " + datainfo + "->data;");
  outstr += prep_str(rows + " = " + datainfo + "->rows;");
  outstr += prep_str(cols + " = " + datainfo + "->cols;");
  //outstr += prep_str(host + " = " + fcnname + "(\"" + fname + "\");");
  outstr += "\n";

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


  headerfile->pushFcnDec(header_outstr);
  cppfile->pushFcnDef(cpp_outstr);
  cudafile->pushMain(outstr);
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
void ParseUtils::writeDatafile(string fname, string object, string type){
  string outstr;

  // define names of vars in prog
  obj_names objnames = get_obj_names(object);
  string host = objnames.host;
  string rows = objnames.rows;
  string cols = objnames.cols;
  string outstream = objnames.outstream;

  outstr += prep_str("// output data to file");
  outstr += prep_str("ofstream " + outstream + ";");
  outstr += prep_str(outstream + ".open(\"" + fname + "\");");
  outstr += "\n";

  outstr += prep_str("// write data out to file");
  outstr += prep_str(outstream + " << \"rows: \" << " + 
      rows + " << \"\\n\";");
  outstr += prep_str(outstream + " << \"cols: \" << " + 
      cols + " << \"\\n\";");

  outstr += prep_str("for (int r=0; r<" + rows + "; r++){"); indent++;
  outstr += prep_str("for (int c=0; c<" + cols + "; c++){"); indent++;
  outstr += prep_str(outstream + " << " + host + 
      "[r*" + cols + "+c] << \" \";"); indent--;
  outstr += prep_str("}"); 
  outstr += prep_str(outstream + " << \"\\n\";"); indent--;
  outstr += prep_str("}");
  outstr += prep_str(outstream + ".close();");
  outstr += "\n";

  cudafile->pushMain(outstr);
}

/********************************
 * Function: mapFcn
 * ----------------
 * Generates code for a map across an object
 */
void ParseUtils::mapFcn(string fcnname, string object, string type, string op, string alter){
  int old_indent = indent; // save indent level
  indent = 0; // since we're defining a function, we start at no indent
  fcnname = numbered_fcnname(fcnname);

  // define names of vars in prog
  obj_names objnames = get_obj_names(object);
  string host = objnames.host;
  string dev = objnames.dev;
  string rows = objnames.rows;
  string cols = objnames.cols;

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


