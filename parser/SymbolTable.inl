#include "SymbolTable.h"

using namespace std;

template <typename TYPE>
void SymbolTable::setData(string name, TYPE* data){

  int type = getTypeInSymtab(name);
  int idx = getIdxInSymtab(name);

  switch (type) {
    case FLOAT:
      float_symtab[idx].data = data;
      break;
    case INT:
      int_symtab[idx].data = data;
      break;
    case STRING:
      string_symtab[idx].data = data;
      break;
  }
}
