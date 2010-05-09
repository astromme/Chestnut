/* Tokenizer for Chestnut Code 
 * ---------------------------
 * Defines all valid tokens in
 * Chestnut code.
 */

%{
#include <stdio.h>
#include <string.h>
#include "parser.hpp"
%}
%%
[0-9]+\.?[0-9]*         yylval.string=strdup(yytext); return NUMBER; // yylval.number=atoi(yytext); return NUMBER;

read                    return TOKREAD; /* read and write */
write                   return TOKWRITE;
print1D                 return TOKPRINT1D;
print                   return TOKPRINT;

timer                   return TOKTIMER; /* timer */
start                   return TOKSTART;
stop                    return TOKSTOP;

foreach                 return TOKFOREACH; /* foreach keywords */

vector                  return TOKVECTOR; /* array vs single elt */
scalar                  return TOKSCALAR;

map                     return TOKMAP; /* kernel functions */
reduce                  return TOKREDUCE;
sort                    return TOKSORT;

int                     yylval.string=strdup(yytext); return TOKINT; /* types */
float                   yylval.string=strdup(yytext); return TOKFLOAT;
string                  yylval.string=strdup(yytext); return TOKSTRING;

[a-zA-Z][a-zA-Z0-9_]*   yylval.string=strdup(yytext); return ID; /* ID's used as variable names */
\".*\"                  yylval.string=strdup(yytext); return FNAME; /* filename, like ID but is between quotes and contains anything */

\"                      return QUOTE;
\(                      return LPAREN;
\)                      return RPAREN;
\{                      return LBRACE;
\}                      return RBRACE;

==                      yylval.string=strdup(yytext); return EQUALS; /* (in)equalities */
!=                      yylval.string=strdup(yytext); return NEQ;
!                       yylval.string=strdup(yytext); return NOT;
\<=                     yylval.string=strdup(yytext); return LTEQ;
\<                      yylval.string=strdup(yytext); return LT;
>=                      yylval.string=strdup(yytext); return GTEQ;
>                       yylval.string=strdup(yytext); return GT;
=                       yylval.string=strdup(yytext); return ASSIGN;

;                       return SEMICOLON; /* separators */
,                       return COMMA;

\+                      yylval.string=strdup(yytext); return PLUS; /* binary ops */
\-                      yylval.string=strdup(yytext); return MINUS;
\*                      yylval.string=strdup(yytext); return MULT;
\/                      yylval.string=strdup(yytext); return DIV;
%                       yylval.string=strdup(yytext); return MOD;
\|\|                    yylval.string=strdup(yytext); return OR;
&&                      yylval.string=strdup(yytext); return AND;

\n                      /* ignore end of line */;
[ \t]+                  /* ignore whitespace */;
\/\/.*[\n]*             /* ignore line */;
%%
