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

  if (type == FLOAT){
    SymbolEntry<float> symentry;
    symentry.name = name;
    symentry.category = category;
    symentry.type = type;
    symentry.scope = scope;
    symentry.data = NULL;
    float_symtab.push_back(symentry);
  } else if (type == INT){
    SymbolEntry<int> symentry;
    symentry.name = name;
    symentry.category = category;
    symentry.type = type;
    symentry.scope = scope;
    symentry.data = NULL;
    int_symtab.push_back(symentry);
  } else if (type == STRING){
    SymbolEntry<string> symentry;
    symentry.name = name;
    symentry.category = category;
    symentry.type = type;
    symentry.scope = scope;
    symentry.data = NULL;
    string_symtab.push_back(symentry);
  } else {
    fprintf(stderr, "in addEntry: type is invalid");
    exit(2);
  }
} 

bool SymbolTable::isInSymtab(string name){
  for (unsigned int i=0; i<float_symtab.size(); i++){
    if (name == float_symtab[i].name){
      return true;
    }
  }
  for (unsigned int i=0; i<int_symtab.size(); i++){
    if (name == int_symtab[i].name){
      return true;
    }
  }
  for (unsigned int i=0; i<string_symtab.size(); i++){
    if (name == string_symtab[i].name){
      return true;
    }
  }
  return false;
}

int SymbolTable::getTypeInSymtab(string name){
  for (unsigned int i=0; i<float_symtab.size(); i++){
    if (name == float_symtab[i].name)
      return float_symtab[i].type;
  }
  for (unsigned int i=0; i<int_symtab.size(); i++){
    if (name == int_symtab[i].name)
      return int_symtab[i].type;
  }
  for (unsigned int i=0; i<string_symtab.size(); i++){
    if (name == string_symtab[i].name)
      return string_symtab[i].type;
  }
  return -1;
}

int SymbolTable::getIdxInSymtab(string name){
  for (unsigned int i=0; i<float_symtab.size(); i++){
    if (name == float_symtab[i].name)
      return i;
  }
  for (unsigned int i=0; i<int_symtab.size(); i++){
    if (name == int_symtab[i].name)
      return i;
  }
  for (unsigned int i=0; i<string_symtab.size(); i++){
    if (name == string_symtab[i].name)
      return i;
  }
  return -1;
}

void SymbolTable::print(){
  printf("[idx]\tname\tcategory\ttype\tscope\n");
  printf("-----\t----\t--------\t----\t-----\n");
  for (unsigned int i=0; i<float_symtab.size(); i++){
    printf("[%d]\t%s\t%d\t\t%d\t%d\n",
        i, float_symtab[i].name.c_str(), float_symtab[i].category, float_symtab[i].type, float_symtab[i].scope);
  }
  for (unsigned int i=0; i<int_symtab.size(); i++){
    printf("[%d]\t%s\t%d\t\t%d\t%d\n",
        i, int_symtab[i].name.c_str(), int_symtab[i].category, int_symtab[i].type, int_symtab[i].scope);
  }
  for (unsigned int i=0; i<string_symtab.size(); i++){
    printf("[%d]\t%s\t%d\t\t%d\t%d\n",
        i, string_symtab[i].name.c_str(), string_symtab[i].category, string_symtab[i].type, string_symtab[i].scope);
  }
}
