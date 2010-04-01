%{
#include <stdio.h>
#include <string.h>
#include <string>
#include <iostream>
#include <vector>

using namespace std;

int yyparse(void);
int yylex(void);

template <typename TYPE>
bool isInVector(vector<TYPE> v, TYPE val);

void yyerror(const char *str)
{
	fprintf(stderr,"error: %s\n",str);
}

int yywrap()
{
	return 1;
}

enum types {
  VARIABLE = 20,
  FUNCTION
}

//struct symentry {
//  char name;
//  int type;
//};
  
vector<string> symtable;

template <typename TYPE>
bool isInVector(vector<TYPE> v, TYPE val){
  for (int i=0; i<v.size(); i++){
    if (v[i] == val){
      return true;
    }
  }
  return false;
}

//bool isNameInSymtable(string name){
//  for (int i=0; i<symtable.size(); i++){
//    if (symtable[i].name == name)
//      return true;
//  }
//  return false;
//}

main()
{
	yyparse();
}

// Global Vars

string heater="NoVal";
string datatype="NoVal"; // could be int, float, etc
string fname="NoVal";


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
%token <string> INT
%token <string> FLOAT
%token <string> STRING
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
	heat_switch | target_set | heater_select | datatype_set | readdata_set | map_call | write_call

write_call:
  TOKWRITE LPAREN WORD COMMA filename RPAREN
  {
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
    fname = $5;

    if (isInVector<string>(symtable, object)) {
      string err = "\"" + object + "\" already in symbol table";
      yyerror(err.c_str());
      exit(2);
    }

    symtable.push_back(object);
    printf("Read >\tObject: %s, Filename: %s\n", object.c_str(), fname.c_str());
    for (int i=0; i<symtable.size(); i++){
      cout << symtable[i] << endl;
    }
  }
;

filename: QUOTE WORD QUOTE
  {
    $$=$2;
  }
;

possible_types: INT | FLOAT | STRING

possible_ops: PLUS | MINUS | MULT | DIV

heat_switch:
	TOKHEAT STATE 
	{
		if($2)
			printf("\tHeater '%s' turned on\n", heater.c_str());
		else
			printf("\tHeat '%s' turned off\n", heater.c_str());
	}
;

target_set:
	TOKTARGET TOKTEMPERATURE NUMBER
	{
		printf("\tHeater '%s' temperature set to %d\n",heater.c_str(), $3);
	}
;

heater_select:
	TOKHEATER WORD
	{
		printf("\tSelected heater '%s'\n",$2);
		heater=$2;
	}
;
