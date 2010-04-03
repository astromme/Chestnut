#ifndef _SYMBOL_ENTRY_H
#define _SYMBOL_ENTRY_H

class SymbolEntry {
  public:
    SymbolEntry() {}; // do nothing
    std::string name;
    int category; // var, fcn, etc
    int type; // int, float, string, etc
    int scope;
};

#endif
