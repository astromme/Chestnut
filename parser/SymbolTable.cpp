#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include "SymbolTable.h"

using namespace std;

SymbolTable::SymbolTable() {}; // do nothing

/********************************
 * Function: addEntry
 * ------------------
 * Add an entry into the symbol table after checking that it's not already in
 * there
 *
 * Inputs:
 *    name: name of function, variable, etc
 *    category: FUNCTION, VARIABLE, etc
 *    type: int, float, string, etc
 *    scope: scoping level. Defaults to zero. We may not run into scoping
 *           issues, but if we do, it's here
 *
 * Returns:
 *    true on success, false on fail. Fails if entry is already in symtab
 */
bool SymbolTable::addEntry(string name, int category, string type, int scope){
  if (isInSymtab(name)){
    return false;
    //fprintf(stderr, "\"%s\" already in symbol table\n", name.c_str());
    //exit(2);
  }

  symbol_entry symentry;
  symentry.name = name;
  symentry.category = category;
  symentry.type = type;
  symentry.scope = scope;
  symtab.push_back(symentry);
  return true;
} 


/********************************
 * Function: isInSymtab
 * --------------------
 * Checks if entry is in symbol table
 *
 * Input:
 *    name: name of function, variable, etc
 */
bool SymbolTable::isInSymtab(string name){
  for (unsigned int i=0; i<symtab.size(); i++){
    if (name == symtab[i].name){
      return true;
    }
  }
  return false;
}

/********************************
 * Function: getTypeInSymtab
 * -------------------------
 * Grabs the type of an entry in the symbol table
 *
 * Input:
 *    name: name of function, variable, etc
 * TODO: throw an exception instead of returning -1
 */
string SymbolTable::getType(string name){
  for (unsigned int i=0; i<symtab.size(); i++){
    if (name == symtab[i].name)
      return symtab[i].type;
  }
  // if we're here, name is not in symtab, so let's throw an error
  char error_msg[100];
  sprintf(error_msg, "in getTypeInSymtab: '%s' not found in SymbolTable", name.c_str());
  throw runtime_error(error_msg);
}


/********************************
 * Function: getIdxInSymtab
 * ------------------------
 * Return index of an entry in symbol table
 *
 * Input:
 *    name: name of function, variable, etc
 */
int SymbolTable::getIdx(string name){
  for (unsigned int i=0; i<symtab.size(); i++){
    if (name == symtab[i].name)
      return i;
  }
  return -1;
}

/********************************
 * Function: print
 * ---------------
 * Prints out entries in symbol table in a human-readable way
 */
void SymbolTable::print(){
  printf("[idx]\tname\tcategory\ttype\tscope\n");
  printf("-----\t----\t--------\t----\t-----\n");
  for (unsigned int i=0; i<symtab.size(); i++){
    printf("[%d]\t%s\t%d\t\t%s\t%d\n",
        i,
        symtab[i].name.c_str(),
        symtab[i].category,
        symtab[i].type.c_str(),
        symtab[i].scope);
  }
}
