#ifndef _FILE_UTILS_H
#define _FILE_UTILS_H

#include <string>
#include <vector>
#include <fstream>

using std::string;
using std::vector;

class FileUtils {
  public:
    FileUtils(string fname); // constructor
    ~FileUtils(); // destructor

    void writeAll();

    void pushInclude(string include);
    void pushMain(string main);
    void pushFcnDec(string fcnDec);
    void pushFcnDef(string fcnDef);

    string fname();

  private:
    std::ofstream outfile;
    string outfname;

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
