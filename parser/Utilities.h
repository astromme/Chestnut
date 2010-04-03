#ifndef _UTILITIES_H
#define _UTILITIES_H

#include <fstream>
#include <string>

class Utilities {
  public:
    Utilities(std::string myoutfile);
    void initialize();
    void finalize();
    void readDatafile(std::string fname, std::string object, std::string type);
    void writeDatafile(std::string fname, std::string object, std::string type);
   
    //template <typename TYPE> TYPE* readDatafile(std::string fname, int *rows, int *cols);
  private:
    std::ofstream outfile;
    int indent; // number of indentations to make

    void add_include(std::string header);
    void write_string(std::string str);
    void write_newline();
    std::string prep_str(std::string str);

  
};

#include "Utilities.inl"

#endif
