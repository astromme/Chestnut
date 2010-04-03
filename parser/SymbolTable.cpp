#include <cstdio>
#include <cstdlib>
#include "SymbolTable.h"

using namespace std;

SymbolTable::SymbolTable() {}; // do nothing

void SymbolTable::addEntry(string name, int category, int type, int scope){
  if (isInSymtab(name)){
    fprintf(stderr, "\"%s\" already in symbol table\n", name.c_str());
    exit(2);
  }

  SymbolEntry symentry;
  symentry.name = name;
  symentry.category = category;
  symentry.type = type;
  symentry.scope = scope;
  symtab.push_back(symentry);
} 

bool SymbolTable::isInSymtab(string name){
  for (unsigned int i=0; i<symtab.size(); i++){
    if (name == symtab[i].name){
      return true;
    }
  }
  return false;
}

int SymbolTable::getTypeInSymtab(string name){
  for (unsigned int i=0; i<symtab.size(); i++){
    if (name == symtab[i].name)
      return symtab[i].type;
  }
  return -1;
}

int SymbolTable::getIdxInSymtab(string name){
  for (unsigned int i=0; i<symtab.size(); i++){
    if (name == symtab[i].name)
      return i;
  }
  return -1;
}

void SymbolTable::print(){
  printf("[idx]\tname\tcategory\ttype\tscope\n");
  printf("-----\t----\t--------\t----\t-----\n");
  for (unsigned int i=0; i<symtab.size(); i++){
    printf("[%d]\t%s\t%d\t\t%d\t%d\n",
        i, symtab[i].name.c_str(), symtab[i].category, symtab[i].type, symtab[i].scope);
  }
}
