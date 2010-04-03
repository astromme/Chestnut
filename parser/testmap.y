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
  utils.initializeIncludes();
  utils.initializeMain();

	yyparse();

  utils.finalizeMain();

  utils.writeBuffer();

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
    utils.writeDatafile(fname, object, datatype);
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

    // FIXME!! -- FLOAT?? really..
    symtab.addEntry(object, VARIABLE, FLOAT);
    utils.readDatafile(fname, object, datatype);
    
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
