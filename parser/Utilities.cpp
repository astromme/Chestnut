#include <iostream>
#include "Utilities.h"

using namespace std;

// constructor
Utilities::Utilities(string outfname) {
  // initialize file so that it's blank (with ios::trunc)
  outfile.open(outfname.c_str(), ios::trunc);

  indent = 0; // initially no indent offset 
} 

// destructor
Utilities::~Utilities(){
  outfile.close(); // close file
}

/********************************
 * Function: writeBuffer
 * ---------------------
 * Writes contents of vectors containing C++ code to the outfile
 */
void Utilities::writeBuffer(){
  // write #includes
  for (unsigned int i=0; i<includeStrs.size(); i++)
    writeString(includeStrs[i]);

  // write function declarations
  for (unsigned int i=0; i<fcnDecStrs.size(); i++)
    writeString(fcnDecStrs[i]);

  // write main()
  for (unsigned int i=0; i<mainStrs.size(); i++)
    writeString(mainStrs[i]);

  // write function definitions
  for (unsigned int i=0; i<fcnDefStrs.size(); i++)
    writeString(fcnDefStrs[i]);
}


/********************************
 * Function: initializeMain
 * ------------------------
 * Consolidates the namespace and beginning of main(). Returns a string which
 * gets put into a vector so we can add more later.
 */
void Utilities::initializeMain(){
  string startmain;
  // use standard namespace (?)
  startmain += "using namespace std;\n\n";
  startmain += "int main() {\n";

  mainStrs.push_back(startmain);
  indent++; // add indentation level
}


/********************************
 * Function: finalizeMain
 * ----------------------
 * We're done with main(), so return a string with the `return 0` standard
 * stuff and a close brace.
 */
void Utilities::finalizeMain(){
  string endmain;
  endmain += "  return 0;\n";
  endmain += "}\n"; // close main()

  mainStrs.push_back(endmain);
}

/********************************
 * Function: initializeIncludes
 * ----------------------------
 * Consolidates some baseline #includes. These are stored as strings and
 * returned to be stored in a vector. The vector is eventually written out to
 * the file, but in case we want to add more #includes later on, we can.
 */
void Utilities::initializeIncludes(){
  // add #include infos
  string includes;
  includes += add_include("string");
  includes += add_include("fstream");
  includes += add_include("iostream");
  includes += "\n";

  includeStrs.push_back(includes);
}


/********************************
 * Function: add_include
 * ---------------------
 * given a header file, wraps it in the customary #include<foo> syntax and
 * returns it.
 */
string Utilities::add_include(string header){
  return "#include <" + header + ">\n";
}

// writes a newline to the outfile
void Utilities::write_newline(){ writeString("\n"); }

// writes given string to the outfile
void Utilities::writeString(string str){ outfile << str; }

/********************************
 * Function: prep_str
 * ------------------
 * prepares a string for file insertion by appending a newline and indenting
 * the proper amount, based on the global indent offset. Returns the indented,
 * "newlined" string
 */
string Utilities::prep_str(string str) { 
  string indentstr = "";
  for (int i=0; i<indent; i++){
    indentstr += "  ";
  }
  return indentstr + str + "\n";
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
void Utilities::readDatafile(string fname, string object, string type){
  string outstr;

  // define names of vars in prog
  string objectstream = object + "_instream";
  string rows = object + "_rows";
  string cols = object + "_cols";
  string garbage = object + "_garbage";

  outstr += prep_str("ifstream " + objectstream + ";");
  outstr += prep_str(objectstream + ".open(\"" + fname + "\");");
  outstr += "\n";

  outstr += prep_str("// read in rows and column data");
  outstr += prep_str("string " + garbage + ";");
  outstr += prep_str("int " + rows + ", " + cols +";");
  outstr += prep_str(objectstream + " >> " + garbage + "; " + objectstream + " >> " + rows + ";");
  outstr += prep_str(objectstream + " >> " + garbage + "; " + objectstream + " >> " + cols + ";");
  outstr += "\n";

  outstr += prep_str("// allocate memory for data");
  outstr += prep_str(type + "** " + object + " = new " + type + "*[" + rows + "];");
  outstr += prep_str("for (int r=0; r< " + rows + "; r++){"); indent++;
  outstr += prep_str(object + "[r] = new " + type + "[" + cols + "];"); indent--;
  outstr += prep_str("}");
  outstr += "\n";

  outstr += prep_str("// read in data from file");
  outstr += prep_str("for (int r=0; r<" + rows + "; r++){"); indent++;
  outstr += prep_str("for (int c=0; c<" + cols + "; c++){"); indent++;
  outstr += prep_str(objectstream + " >> " + object + "[r][c];"); indent--;
  outstr += prep_str("}"); indent--;
  outstr += prep_str("}");
  outstr += prep_str(objectstream + ".close();");
  outstr += "\n";

  mainStrs.push_back(outstr);
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
void Utilities::writeDatafile(string fname, string object, string type){
  string outstr;

  // define names of vars in prog
  string objectstream = object+"_outstream";
  string rows = object + "_rows";
  string cols = object + "_cols";

  outstr += prep_str("// output data to file");
  outstr += prep_str("ofstream " + objectstream + ";");
  outstr += prep_str(objectstream + ".open(\"" + fname + "\");");
  outstr += "\n";

  outstr += prep_str("// write data out to file");
  outstr += prep_str(objectstream + " << \"rows: \" << " + 
      rows + " << \"\\n\";");
  outstr += prep_str(objectstream + " << \"cols: \" << " + 
      cols + " << \"\\n\";");

  outstr += prep_str("for (int r=0; r<" + rows + "; r++){"); indent++;
  outstr += prep_str("for (int c=0; c<" + cols + "; c++){"); indent++;
  outstr += prep_str(objectstream + " << " + object + 
      "[r][c] << \" \";"); indent--;
  outstr += prep_str("}"); 
  outstr += prep_str(objectstream + " << \"\\n\";"); indent--;
  outstr += prep_str("}");
  outstr += prep_str(objectstream + ".close();");
  outstr += "\n";

  mainStrs.push_back(outstr);
}

/********************************
 * Function: mapFcn
 */
//string Utilities::mapFcn(string
