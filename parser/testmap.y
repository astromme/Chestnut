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

ParseUtils parseutils("cudalang");


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
  parseutils.initializeIncludes();
  parseutils.initializeMain();
  parseutils.initializeHeader();

	yyparse();

  parseutils.finalizeMain();
  parseutils.finalizeHeader();


  parseutils.writeAllFiles();

  //symtab.print();

  return 0;
}

// Global Vars

string datatype="NoVal"; // could be int, float, etc

%}

%token TOKHEATER TOKHEAT TOKTARGET TOKTEMPERATURE 
TOKTYPE TOKREAD TOKMAP TOKWRITE TOKFOR
EQUALS ASSIGN SEMICOLON LPAREN RPAREN LBRACE RBRACE QUOTE COMMA
LT LTEQ GT GTEQ NOT NEQ OR AND

%union 
{
	int number;
	char *string;
}

%token <number> STATE
%token <number> NUMBER
%token <string> ID
%token <string> TOKINT
%token <string> TOKFLOAT
%token <string> TOKSTRING
%token <string> TOKCPP
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
	datatype_set | map_call | write_call | expr | forloop
	//datatype_set | cppdata_set | readdata_set | map_call | write_call | expr

write_call:
  TOKWRITE LPAREN ID COMMA filename RPAREN
  {
    string object = $3;
    string fname = $5;
    parseutils.writeDatafile(fname, object, datatype);
    printf("Write >\tObject: %s, OutFile: %s\n", $3, $5);
  }
;

map_call:
  TOKMAP LPAREN possible_ops COMMA alphanumeric COMMA ID RPAREN
  {
    string op = $3;
    string alter = $5;
    string object = $7;

    string fcnname = "map";

    //symtab.addEntry(fcnname, FUNCTION, FLOAT); // FIXME: FLOAT issue..
    parseutils.mapFcn(fcnname, object, datatype, op, alter);
    printf("Map >\tOperation: %s, Number: %s, Object: %s\n", $3, $5, $7);
    delete $5;
  }
;

alphanumeric: 
  ID | NUMBER
  {
    // need to convert ID | NUMBER to a string
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

// FIXME TODO
// commented region causes shift/reduce because it starts with ID, which could
// also be a valid `expr`
/*
cppdata_set:
  ID ASSIGN TOKCPP
  {
    //printf("cpp: %s\n", $3);
    string object = $1;
    string code = $3;
    parseutils.insertVerbatim(object, code, datatype);
  }
;

readdata_set:
  ID ASSIGN TOKREAD LPAREN filename RPAREN
  {
    string object = $1;
    string fname = $5;

    // FIXME!! -- FLOAT?? really..
    //symtab.addEntry(object, VARIABLE, FLOAT);
    parseutils.readDatafile(fname, object, datatype);
    
    printf("Read >\tObject: %s, Filename: %s\n", object.c_str(), fname.c_str());
  }
;
*/

forloop:
  TOKFOR LPAREN expr SEMICOLON expr SEMICOLON expr RPAREN LBRACE RBRACE
  {
    printf("found for loop\n");
  }
;

filename: QUOTE ID QUOTE
  {
    $$=$2;
  }
;

possible_types: TOKINT | TOKFLOAT | TOKSTRING

possible_ops: PLUS | MINUS | MULT | DIV

/* Expressions! */

expr:
  expr1 exprP
  {
    printf("Found expr\n");
  }
; 

exprP:
  | ASSIGN expr1 exprP
  {
    printf("Found exprP\n");
  }
; 

expr1:
  expr2 expr1P
  {
    printf("Found expr1\n");
  }
; 

expr1P:
  | OR expr2 expr1P
  {
    printf("Found expr1P\n");
  }
; 

expr2:
  expr3 expr2P
  {
    printf("Found expr2\n");
  }
; 

expr2P:
  | AND expr3 expr2P
  {
    printf("Found expr2P\n");
  }
; 

expr3:
  expr4 expr3P
  {
    printf("Found expr3\n");
  }
; 

expr3P:
  | EQUALS expr4 expr3P
  | NEQ expr4 expr3P
  {
    printf("Found expr3P\n");
  }
; 

expr4:
  expr5 expr4P
  {
    printf("Found expr4\n");
  }
; 

expr4P:
  | LT expr5 expr4P
  | LTEQ expr5 expr4P
  | GT expr5 expr4P
  | GTEQ expr5 expr4P
  {
    printf("Found expr4P\n");
  }
; 

expr5:
  expr6 expr5P
  {
    printf("Found expr5\n");
  }
; 

expr5P:
  | PLUS expr6 expr5P
  | MINUS expr6 expr5P
  {
    printf("Found expr5P\n");
  }
; 

expr6:
  expr7 expr6P
  {
    printf("Found expr6\n");
  }
; 

expr6P:
  | MULT expr7 expr6P
  | DIV expr7 expr6P
  {
    printf("Found expr6P\n");
  }
; 

expr7:
    NOT expr7
  | MINUS expr7
  | expr8
  {
    printf("Found expr7\n");
  }
;

expr8:
    LPAREN expr RPAREN
    | ID
    | NUMBER 
  {
    printf("Found expr8\n");
  }
;
