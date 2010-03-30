%{
#include <stdio.h>
#include <string.h>
#include "y.tab.h"
%}
%%
[0-9]+       yylval.number=atoi(yytext); return NUMBER;
heater			 return TOKHEATER;
heat         return TOKHEAT;
type         return TOKTYPE;
read         return TOKREAD;
map          return TOKMAP;
write        return TOKWRITE;
int          yylval.string=strdup(yytext); return INT;
float        yylval.string=strdup(yytext); return FLOAT;
string       yylval.string=strdup(yytext); return STRING;
on|off       yylval.number=!strcmp(yytext,"on"); return STATE;
target       return TOKTARGET;
temperature  return TOKTEMPERATURE;
[a-z0-9_]+		 yylval.string=strdup(yytext);return WORD;
\"           return QUOTE;
\(           return LPAREN;
\)           return RPAREN;
==           return EQUALS;
=            return ASSIGN;
;            return SEMICOLON;
,            return COMMA;
\+           yylval.string=strdup(yytext); return PLUS;
\-           yylval.string=strdup(yytext); return MINUS;
\*           yylval.string=strdup(yytext); return MULT;
\/           yylval.string=strdup(yytext); return DIV;
\n           /* ignore end of line */;
[ \t]+       /* ignore whitespace */;
\/\/.*\n     /* ignore line */;
%%
