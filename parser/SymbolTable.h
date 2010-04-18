#ifndef _SYMBOL_TABLE_H
#define _SYMBOL_TABLE_H

#include <string>
#include <vector>

using std::string;

enum categories {
  VARIABLE_SCALAR = 20,
  VARIABLE_VECTOR,
  FUNCTION
};

enum types {
  FLOAT = 30,
  INT,
  STRING
};

struct symbol_entry {
  string name;
  int category; // var, fcn, etc
  string type; // int, float, string, etc
  int scope;
};

class SymbolTable{
  public:
    SymbolTable();

    // manipulators
    bool addEntry(string name, int category, string type, int scope=0);

    // accessors 
    bool isInSymtab(string name);
    string getType(string name);
    int getCategory(string name);
    int getIdx(string name);
    void print();

  private:
    std::vector<symbol_entry> symtab;

};

#endif
