#ifndef _UTILITIES_H
#define _UTILITIES_H

#include <fstream>
#include <string>

using std::string;

class Utilities {
  public:
    Utilities(string myoutfile);
    ~Utilities();
    string initializeIncludes();
    string initializeMain();
    string finalizeMain();
    void writeString(string str);

    string readDatafile(string fname, string object, string type);
    string writeDatafile(string fname, string object, string type);
  
  private:
    std::ofstream outfile;
    int indent; // number of indentations to make

    string add_include(string header);
    void write_newline();
    string prep_str(string str);
  
};

#include "Utilities.inl"

#endif
