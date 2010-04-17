%{
#include <stdio.h>
#include <string.h>
#include "parser.hpp"
%}
%%
STARTCPP(.*[ \t\n])*ENDCPP yylval.string=strdup(yytext); return TOKCPP;
[0-9]+       yylval.number=atoi(yytext); return NUMBER;
type         return TOKTYPE;
read         return TOKREAD;
map          return TOKMAP;
write        return TOKWRITE;
for          return TOKFOR;
int          yylval.string=strdup(yytext); return TOKINT;
float        yylval.string=strdup(yytext); return TOKFLOAT;
string       yylval.string=strdup(yytext); return TOKSTRING;
[a-zA-Z][a-zA-Z0-9_]*	 yylval.string=strdup(yytext); return ID;
\"           return QUOTE;
\(           return LPAREN;
\)           return RPAREN;
\{           return LBRACE;
\}           return RBRACE;
==           return EQUALS;
!=           return NEQ;
!            return NOT;
\<=          return LTEQ;
\<           return LT;
>=           return GTEQ;
>            return GT;
=            return ASSIGN;
\|\|         return OR;
&&           return AND;
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
