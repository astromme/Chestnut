%{
#include <stdio.h>
#include <string.h>
#include "parser.hpp"
%}
%%
[0-9]+       yylval.number=atoi(yytext); return NUMBER;
type         return TOKTYPE;
read         return TOKREAD;
map          return TOKMAP;
write        return TOKWRITE;
int          yylval.string=strdup(yytext); return TOKINT;
float        yylval.string=strdup(yytext); return TOKFLOAT;
string       yylval.string=strdup(yytext); return TOKSTRING;
[a-z0-9_]+	 yylval.string=strdup(yytext); return WORD;
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
