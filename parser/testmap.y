%{
#include <stdio.h>
#include <string.h>

void yyerror(const char *str)
{
	fprintf(stderr,"error: %s\n",str);
}

int yywrap()
{
	return 1;
}

main()
{
	yyparse();
}

char *heater="NoVal";
char *datatype="NoVal"; // could be int, float, etc
char *object="NoVal";
char *fname="NoVal";

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
    printf("Type >\tSet to %s\n", datatype);
  }
;

readdata_set:
  WORD ASSIGN TOKREAD LPAREN filename RPAREN
  {
    object = $1;
    fname = $5;
    printf("Read >\tObject: %s, Filename: %s\n", object, fname);
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
			printf("\tHeater '%s' turned on\n", heater);
		else
			printf("\tHeat '%s' turned off\n", heater);
	}
;

target_set:
	TOKTARGET TOKTEMPERATURE NUMBER
	{
		printf("\tHeater '%s' temperature set to %d\n",heater, $3);
	}
;

heater_select:
	TOKHEATER WORD
	{
		printf("\tSelected heater '%s'\n",$2);
		heater=$2;
	}
;
