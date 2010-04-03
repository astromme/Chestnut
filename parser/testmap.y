%{
#include <stdio.h>
#include <string.h>
#include <string>
#include <iostream>
#include <vector>
#include "SymbolTable.h"
#include "Utilities.h"

using namespace std;

int yyparse(void);
int yylex(void);

SymbolTable symtab;
Utilities utils("cudalang.cpp");

/***********************************************
 * A Note about these vectors: 
 * 
 * Each of these vectors hold strings that will eventually be written out to a
 * file specified by the constructor parameter of utils in main() of this file.
 * We store these in vectors instead of directly writing them to disk because we
 * might want to add statements to various vectors, and if everything is already
 * written to the file, we'd have a real mess on our hands. For example, we might
 * add more `#include' statements to the includeStrs vector, but if we already
 * started writing main() then we'd be in a pretty pickle.
 */
vector<string> includeStrs;   // #includes
vector<string> mainStrs;      // main function
vector<string> fcnDecStrs;    // function declarations
vector<string> fcnDefStrs;    // function definitions

void yyerror(const char *str)
{
	fprintf(stderr,"error: %s\n",str);
}

int yywrap()
{
	return 1;
}

int main()
{
  includeStrs.push_back(utils.initializeIncludes());
  mainStrs.push_back(utils.initializeMain());
	yyparse();
  mainStrs.push_back(utils.finalizeMain());

  // write out all vectors, in order, to file
  for (unsigned int i=0; i<includeStrs.size(); i++)
    utils.writeString(includeStrs[i]);
  for (unsigned int i=0; i<fcnDecStrs.size(); i++)
    utils.writeString(fcnDecStrs[i]);
  for (unsigned int i=0; i<mainStrs.size(); i++)
    utils.writeString(mainStrs[i]);
  for (unsigned int i=0; i<fcnDefStrs.size(); i++)
    utils.writeString(fcnDefStrs[i]);

  return 0;
}

// Global Vars

string datatype="NoVal"; // could be int, float, etc

%}

%token TOKHEATER TOKHEAT TOKTARGET TOKTEMPERATURE 
TOKTYPE TOKREAD TOKMAP TOKWRITE
EQUALS ASSIGN SEMICOLON LPAREN RPAREN QUOTE COMMA

%union 
{
	int number;
	char *string;
}

%token <number> STATE
%token <number> NUMBER
%token <string> WORD
%token <string> TOKINT
%token <string> TOKFLOAT
%token <string> TOKSTRING
%token <string> PLUS
%token <string> MINUS
%token <string> MULT
%token <string> DIV

%type <string> possible_types
%type <string> possible_ops
%type <string> filename

%%

commands: /* empty */
	| commands command SEMICOLON
;

command:
	datatype_set | readdata_set | map_call | write_call

write_call:
  TOKWRITE LPAREN WORD COMMA filename RPAREN
  {
    string object = $3;
    string fname = $5;
    string writedata = utils.writeDatafile(fname, object, datatype);
    mainStrs.push_back(writedata);
    printf("Write >\tObject: %s, OutFile: %s\n", $3, $5);
  }
;

map_call:
  TOKMAP LPAREN possible_ops COMMA NUMBER COMMA WORD RPAREN
  {
    printf("Map >\tOperation: %s, Number: %d, Object: %s\n", $3, $5, $7);
  }
;

datatype_set:
  TOKTYPE ASSIGN possible_types
  {
    datatype=$3;
    printf("Type >\tSet to %s\n", datatype.c_str());
  }
;

readdata_set:
  WORD ASSIGN TOKREAD LPAREN filename RPAREN
  {
    string object = $1;
    string fname = $5;

    int rows, cols;
    symtab.addEntry(object, VARIABLE, FLOAT);
    string readdata = utils.readDatafile(fname, object, datatype);
    mainStrs.push_back(readdata);
    
    printf("Read >\tObject: %s, Filename: %s\n", object.c_str(), fname.c_str());
    symtab.print();
  }
;

filename: QUOTE WORD QUOTE
  {
    $$=$2;
  }
;

possible_types: TOKINT | TOKFLOAT | TOKSTRING

possible_ops: PLUS | MINUS | MULT | DIV
