#ifndef _PARSE_UTILS_H
#define _PARSE_UTILS_H

#include <fstream>
#include <string>
#include <vector>
#include "SymbolTable.h"
#include "FileUtils.h"

using std::vector;
using std::string;

typedef struct {
  string host;
  string dev;
  string rows;
  string cols;
  string datainfo;
  string instream;
  string outstream;
  string garbage;
  string timerstart;
  string timerstop;
} obj_names;

class ParseUtils {
  public:
    ParseUtils(string myoutfile);
    ~ParseUtils();

    void initializeIncludes();
    void initializeMain();
    void initializeHeader();

    void finalizeMain();
    void finalizeHeader();

    void writeCuda(string str);
    void writeAllFiles();
    //void writeBuffer();

    void insertVerbatim(string object, string code, string type);
    void makeForeach(string object, string type, string datarows, string datacols, string expr);
    void makeVector(string object, string type);
    void makeScalar(string object, string type);
    
    void makeTimer(string timer);
    void makeTimerStart(string timer);
    void makeTimerStop(string timer);

    void makeMap(string source, string destination, string op, string modify);
    void makeReduce(string source, string destination, string op);
    void makeSort(string source, string destination, string comparator);

    void makeReadDatafile(string fname, string object, string type);
    void makeWriteDatafile(string fname, string object);

    void makePrintData(string object);
    void makePrintData1D(string object);
  
  private:
    //std::ofstream cudafile;
    FileUtils *cudafile, *cppfile, *headerfile;
    int indent; // number of indentations to make
    int fcncount; // number of functions instantiated so far

    bool parselibs_included;
    bool cudalang_included;
    bool rand_included; // #includes and srand(time(NULL)) conditions satisfied

    string processForeachExpr(string expr, const obj_names &objnames);
    string numberedId(string fcnname);
    string getThrustOp(string op, string type);
    string copyDevData(string source, string destination);

    SymbolTable symtab; // symbol table

    string add_include(string header);
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
