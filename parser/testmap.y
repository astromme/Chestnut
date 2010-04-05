%{
#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
#include <vector>
#include <sstream>
#include "SymbolTable.h"
#include "ParseUtils.h"

using namespace std;

int yyparse(void);
int yylex(void);

SymbolTable symtab;
ParseUtils parseutils("cudalang");
int fcncount; // number of functions instantiated so far


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
  fcncount = 0;
  parseutils.initializeIncludes();
  parseutils.initializeMain();

	yyparse();

  parseutils.finalizeMain();
  parseutils.writeBuffer();

  symtab.print();

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
%type <string> alphanumeric

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
    parseutils.writeDatafile(fname, object, datatype);
    printf("Write >\tObject: %s, OutFile: %s\n", $3, $5);
  }
;

map_call:
  TOKMAP LPAREN possible_ops COMMA alphanumeric COMMA WORD RPAREN
  {
    string op = $3;
    string alter = $5;
    string object = $7;


    stringstream ss; ss << fcncount;
    fcncount++;
    string fcnname = "map" + ss.str();

    symtab.addEntry(fcnname, FUNCTION, FLOAT); // FIXME: FLOAT issue..
    parseutils.mapFcn(fcnname, object, datatype, op, alter);
    printf("Map >\tOperation: %s, Number: %s, Object: %s\n", $3, $5, $7);
    delete $5;
  }
;

alphanumeric: 
  WORD | NUMBER
  {
    // need to convert WORD | NUMBER to a string
    stringstream ss; ss << $1;
    string str = ss.str();
    char* chstr = new char[str.length()+1];
    strcpy (chstr, str.c_str());
    $$ = chstr;
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
    parseutils.readDatafile(fname, object, datatype);
    
    printf("Read >\tObject: %s, Filename: %s\n", object.c_str(), fname.c_str());
  }
;

filename: QUOTE WORD QUOTE
  {
    $$=$2;
  }
;

possible_types: TOKINT | TOKFLOAT | TOKSTRING

possible_ops: PLUS | MINUS | MULT | DIV
