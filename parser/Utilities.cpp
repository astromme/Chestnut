#include <iostream>
#include "Utilities.h"

using namespace std;

Utilities::Utilities(string outfname) {
  // initialize file so that it's blank (ios::trunc)
  outfile.open(outfname.c_str(), ios::trunc);

  indent = 0; // initially no indent offset 
} 

void Utilities::initialize(){
  // add #include infos
  add_include("string");
  add_include("fstream");
  add_include("iostream");
  write_newline();

  // use standard namespace (?)
  write_string("using namespace std;\n");
  write_newline();

  write_string("int main() {\n"); indent++;
}

void Utilities::finalize(){
  write_string("}\n"); // close main()

  outfile.close(); // close file
}

void Utilities::add_include(string header){
  write_string("#include <" + header + ">\n");
}

void Utilities::write_newline(){ write_string("\n"); }

void Utilities::write_string(string str){ outfile << str; }

// prepare a string for file insertion by appending a newline and indenting
// the proper amount, based on the indent offset
string Utilities::prep_str(string str) { 
  string indentstr = "";
  for (int i=0; i<indent; i++){
    indentstr += "  ";
  }

  return indentstr + str + "\n";
}

void Utilities::readDatafile(string fname, string object, string type){
  string outstr;
  string objectstream = object+"_instream";

  outstr += prep_str("ifstream " + objectstream + ";");
  outstr += prep_str(objectstream + ".open(\"" + fname + "\");");
  outstr += prep_str(""); // equivalent to newline

  outstr += prep_str("// read in rows and column data");
  outstr += prep_str("string garbage;");
  outstr += prep_str("int rows, cols;");
  outstr += prep_str(objectstream + " >> garbage; " + objectstream + " >> rows;");
  outstr += prep_str(objectstream + " >> garbage; " + objectstream + " >> cols;");

  outstr += prep_str("");
  outstr += prep_str("// allocate memory for data");
  outstr += prep_str(type + "** " + object + " = new " + type + "*[rows];");
  outstr += prep_str("for (int r=0; r<rows; r++){"); indent++;
  outstr += prep_str(object + "[r] = new " + type + "[cols];"); indent--;
  outstr += prep_str("}");

  outstr += prep_str("");
  outstr += prep_str("// read in data from file");
  outstr += prep_str("for (int r=0; r<rows; r++){"); indent++;
  outstr += prep_str("for (int c=0; c<cols; c++){"); indent++;
  outstr += prep_str(objectstream + " >> " + object + "[r][c];"); indent--;
  outstr += prep_str("}"); indent--;
  outstr += prep_str("}");
  outstr += prep_str("");

  write_string(outstr);
}

void Utilities::writeDatafile(string fname, string object, string type){
  string outstr;
  string objectstream = object+"_outstream";

  outstr += prep_str("// output data to file");
  outstr += prep_str("ofstream " + objectstream + ";");
  outstr += prep_str(objectstream + ".open(\"" + fname + "\");");
  outstr += prep_str(""); // equivalent to newline

  outstr += prep_str("// write data out to file");
  outstr += prep_str(objectstream + " << \"rows: \";");
  outstr += prep_str(objectstream + " << rows;");
  outstr += prep_str(objectstream + " << \"\\n\";");
  outstr += prep_str(objectstream + " << \"cols: \";");
  outstr += prep_str(objectstream + " << cols;");
  outstr += prep_str(objectstream + " << \"\\n\";");

  outstr += prep_str("for (int r=0; r<rows; r++){"); indent++;
  outstr += prep_str("for (int c=0; c<cols; c++){"); indent++;
  outstr += prep_str(objectstream + " << " + object + "[r][c];");
  outstr += prep_str(objectstream + " << \" \";"); indent--;
  outstr += prep_str("}"); indent--;
  outstr += prep_str(objectstream + " << \"\\n\";");
  outstr += prep_str("}");

  write_string(outstr);
}
