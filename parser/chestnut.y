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

ParseUtils parseutils("chestnut");

struct vardec_helper {
  int op;
  string fname;
  string rows;
  string cols;
  string expr;
};
vardec_helper curVarDec;

enum vardec_ops {
  VARDEC_NOOP = 10,
  VARDEC_READ,
  VARDEC_FOREACH,
  VARDEC_VECTOR,
  VARDEC_SCALAR
};

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

  //parseutils.initializeHeader(); TODO remove

  yyparse();

  parseutils.finalizeMain();
  // parseutils.finalizeHeader(); TODO remove


  parseutils.writeAllFiles();

  //symtab.print();

  return 0;
}

// Global Vars

string datatype="NoVal"; // could be int, float, etc

%}

%token 
TOKREAD TOKWRITE TOKPRINT TOKPRINT1D 
TOKTIMER TOKSTART TOKSTOP
TOKMAP TOKREDUCE TOKSORT
TOKFOREACH
TOKVECTOR TOKSCALAR
EQUALS SEMICOLON LPAREN RPAREN LBRACE RBRACE QUOTE COMMA
TOKROW TOKCOL TOKMAXROWS TOKMAXCOLS


%union 
{
	int number;
	char *string;
}

%token <number> STATE
%token <string> NUMBER
%token <string> ID
%token <string> TOKINT
%token <string> TOKFLOAT
%token <string> TOKSTRING
%token <string> TOKCPP
%token <string> ASSIGN
%token <string> PLUS
%token <string> MINUS
%token <string> MULT
%token <string> DIV
%token <string> MOD
%token <string> LT 
%token <string> LTEQ
%token <string> GT 
%token <string> GTEQ 
%token <string> NOT 
%token <string> NEQ 
%token <string> OR 
%token <string> AND

%type <string> type
%type <string> op
%type <string> filename
%type <string> alphanumeric
%type <string> read_data
%type <string> expr
%type <string> sort_comparators
//%type <string> map_op_object
//%type <string> foreach_contents

%right ASSIGN
%left PLUS MINUS
%left MULT DIV MOD

%%

commands: /* empty */
	| commands command SEMICOLON
;

command:
  map_call | reduce_call | sort_call | write_call | variable_declarations | print | timer_start | timer_stop

variable_declarations:
  type ID variable_declarations_end
  {
    string type = $1;
    string name = $2;
    string fname, rows, cols, expr;

    // read was chosen
    switch (curVarDec.op){
      case VARDEC_READ:
        fname = curVarDec.fname;
        parseutils.makeReadDatafile(fname, name, type);
        break;
        
      case VARDEC_FOREACH:
        rows = curVarDec.rows;
        cols = curVarDec.cols;
        expr = curVarDec.expr;

        parseutils.makeForeach(name, type, rows, cols, expr);
        break;
    
      case VARDEC_VECTOR: 
        parseutils.makeVector(name, type);
        break;
      case VARDEC_SCALAR:
        parseutils.makeScalar(name, type);
        break;
        
    }
  
    printf("variable declaration %s %s\n", type.c_str(), name.c_str());
    free($1); free($2);
  }
  | TOKTIMER ID
  {
    string timer = $2;
    parseutils.makeTimer(timer);
    printf("timer declared: %s\n",timer.c_str());
    free($2);
  }


variable_declarations_end: 
  /* empty */ { curVarDec.op = VARDEC_NOOP; }
  | TOKVECTOR { curVarDec.op = VARDEC_VECTOR; }
  | TOKSCALAR { curVarDec.op = VARDEC_SCALAR; }
  | read_data
  | foreach

read_data:
  TOKREAD LPAREN filename RPAREN
  {
    curVarDec.op = VARDEC_READ;
    curVarDec.fname = $3;
    free($3);
  }

foreach:
  NUMBER NUMBER TOKFOREACH LPAREN expr RPAREN
  //NUMBER NUMBER TOKFOREACH LPAREN foreach_contents RPAREN
  {
    curVarDec.op = VARDEC_FOREACH;
    curVarDec.rows = $1;
    curVarDec.cols = $2;
    curVarDec.expr = $5;
    free($1); free($2);
    delete [] $5;
  }

/*foreach_contents: TOKVALUE ASSIGN expr
  {
    $$=$3;
    free($2); 
  }
  */

write_call:
  TOKWRITE LPAREN ID COMMA filename RPAREN
  {
    string object = $3;
    string fname = $5;
    parseutils.makeWriteDatafile(fname, object);
    printf("Write >\tObject: %s, OutFile: %s\n", $3, $5);
    free($3); free($5);
  }

map_call:
  ID ASSIGN TOKMAP LPAREN op COMMA alphanumeric COMMA ID RPAREN
  {
    string op = $5;
    string alter = $7;
    string source = $9;
    string destination = $1;

    string fcnname = "map";

    //symtab.addEntry(fcnname, FUNCTION, FLOAT); // FIXME: FLOAT issue..
    parseutils.makeMap(source, destination, op, alter);
    printf("Map >\tOperation: %s, Number: %s, Object: %s\n", $5, $7, $9);
    free($1); free($5); free($7); free($9);
  }

reduce_call:
  ID ASSIGN TOKREDUCE LPAREN op COMMA ID RPAREN
  {
    string source = $7;
    string destination = $1;
    string op = $5;

    parseutils.makeReduce(source, destination, op);

    free($1); free($5); free($7);
  }

sort_call:
  ID ASSIGN TOKSORT LPAREN sort_comparators COMMA ID RPAREN
  {
    string source = $7;
    string destination = $1;
    string comparator = $5;

    parseutils.makeSort(source, destination, comparator);
    free($1); free($5); free($7);
  }

sort_comparators:
  LT | LTEQ | GT | GTEQ
  {
    $$=$1;
  }

print: printdata | printdata1D

printdata:
  TOKPRINT ID
  {
    string object = $2;
    parseutils.makePrintData(object);
    free($2);
  }

printdata1D:
  TOKPRINT1D ID
  {
    string object = $2;
    parseutils.makePrintData1D(object);
    free($2);
  }

timer_start:
  ID TOKSTART
  {
    string timer = $1;
    parseutils.makeTimerStart(timer);
    printf("timer start: %s\n",timer.c_str());
    free($1);
  }
  
timer_stop:
  ID TOKSTOP
  {
    string timer = $1;
    parseutils.makeTimerStop(timer);
    printf("timer stop: %s\n",timer.c_str());
    free($1);
  }

/*
map_op_object:
    type NUMBER
{ 
  string type=$1, id=$2;
  char *str = new char[type.length()+id.length()+2];
  strcpy(str,$1);
  strcat(str," ");
  strcat(str,$2);
  free($1); free($2);
  $$=str;
}
  | ID
{ 
  $$=$1;
}*/

alphanumeric: 
  ID | NUMBER
  {
    // need to convert ID | NUMBER to a string
    stringstream ss; ss << $1;
    string str = ss.str();
    char* chstr = new char[str.length()+1];
    strcpy (chstr, str.c_str());
    $$ = chstr;
    free($1);
  }

/*
forloop:
  TOKFOR LPAREN expr SEMICOLON expr SEMICOLON expr RPAREN LBRACE RBRACE
  {
    printf("found for loop\n");
  }
;*/

filename: QUOTE ID QUOTE
  {
    $$=$2;
  }

type: TOKINT | TOKFLOAT | TOKSTRING

op: PLUS | MINUS | MULT | DIV

/* Expressions! */
expr:
    expr ASSIGN expr
{ 
  string pt1=$1, pt2=$2, pt3=$3;
  char *str = new char[pt1.length()+pt2.length()+pt3.length()+1];
  strcpy(str,$1);
  strcat(str,$2);
  strcat(str,$3);
  free($1); free($2); free($3);
  $$=str;
}
  | expr PLUS expr
{ 
  string pt1=$1, pt2=$2, pt3=$3;
  char *str = new char[pt1.length()+pt2.length()+pt3.length()+1];
  strcpy(str,$1);
  strcat(str,$2);
  strcat(str,$3);
  free($1); free($2); free($3);
  $$=str;
}
  | expr MINUS expr
{ 
  string pt1=$1, pt2=$2, pt3=$3;
  char *str = new char[pt1.length()+pt2.length()+pt3.length()+1];
  strcpy(str,$1);
  strcat(str,$2);
  strcat(str,$3);
  free($1); free($2); free($3);
  $$=str;
}
  | expr MULT expr
{ 
  string pt1=$1, pt2=$2, pt3=$3;
  char *str = new char[pt1.length()+pt2.length()+pt3.length()+1];
  strcpy(str,$1);
  strcat(str,$2);
  strcat(str,$3);
  free($1); free($2); free($3);
  $$=str;
}
  | expr DIV expr
{ 
  string pt1=$1, pt2=$2, pt3=$3;
  char *str = new char[pt1.length()+pt2.length()+pt3.length()+1];
  strcpy(str,$1);
  strcat(str,$2);
  strcat(str,$3);
  free($1); free($2); free($3);
  $$=str;
}
  | expr MOD expr
{ 
  string pt1=$1, pt2=$2, pt3=$3;
  char *str = new char[pt1.length()+pt2.length()+pt3.length()+1];
  strcpy(str,$1);
  strcat(str,$2);
  strcat(str,$3);
  free($1); free($2); free($3);
  $$=str;
}
  | LPAREN expr RPAREN
{
  string pt1="(", pt2=$2, pt3=")";
  char *str = new char[pt1.length()+pt2.length()+pt3.length()+1];
  strcpy(str,"(");
  strcat(str,$2);
  strcat(str,")");
  free($2);
  $$=str;
}
  | NUMBER { $$=$1; }
  | ID { $$=$1; }
