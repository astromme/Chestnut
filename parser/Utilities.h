#ifndef _UTILITIES_H
#define _UTILITIES_H

#include <fstream>
#include <string>
#include <vector>

using std::vector;
using std::string;

typedef struct {
  string host;
  string dev;
  string rows;
  string cols;
  string instream;
  string outstream;
  string garbage;
} obj_names;

class Utilities {
  public:
    Utilities(string myoutfile);
    ~Utilities();
    void initializeIncludes();
    void initializeMain();
    void finalizeMain();
    void writeString(string str);
    void writeBuffer();

    void readDatafile(string fname, string object, string type);
    void writeDatafile(string fname, string object, string type);
    void mapFcn(string fcnname, string object, string type, string op, string alter);
  
  private:
    std::ofstream outfile;
    int indent; // number of indentations to make

    string add_include(string header);
    void write_newline();
    string prep_str(string str);
    string cudamemcpy_str(string to, string from, string size, string directive);
    obj_names get_obj_names(string obj);

    /***********************************************
     * A Note about these vectors: 
     * 
     * Each of these vectors hold strings that will eventually be written out
     * to a file specified by the constructor parameter of utils in main() of
     * the .y file.  We store these in vectors instead of directly writing
     * them to disk because we might want to add statements to various
     * vectors, and if everything is already written to the file, we'd have a
     * real mess on our hands. For example, we might add more `#include'
     * statements to the includeStrs vector, but if we already started writing
     * main() then we'd be in a pretty pickle.
     */
    vector<string> includeStrs;   // #includes
    vector<string> mainStrs;      // main function
    vector<string> fcnDecStrs;    // function declarations
    vector<string> fcnDefStrs;    // function definitions
  
};

#endif
