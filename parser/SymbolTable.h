#ifndef _SYMBOL_TABLE_H
#define _SYMBOL_TABLE_H

#include <string>
#include <vector>
#include "SymbolEntry.h"

enum categories {
  VARIABLE = 20,
  FUNCTION
};

enum types {
  FLOAT = 30,
  INT,
  STRING
};

struct symbol_entry {
  std::string name;
  int category; // var, fcn, etc
  int type; // int, float, string, etc
  int scope;
};

class SymbolTable{
  public:
    SymbolTable();

    // manipulators
    void addEntry(std::string name, int category, int type, int scope=0);
    template <typename TYPE> void setData(std::string name, TYPE* data);


    // accessors 
    bool isInSymtab(std::string name);
    int getTypeInSymtab(std::string name);
    int getIdxInSymtab(std::string name);
    void print();

  private:
    std::vector<SymbolEntry<float> > float_symtab;
    std::vector<SymbolEntry<int> > int_symtab;
    std::vector<SymbolEntry<std::string> > string_symtab;

};

#include "SymbolTable.inl"

#endif
