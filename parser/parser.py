#!/usr/bin/env python

import re
from nodes import *

identifier = Token('[a-zA-Z][a-zA-Z0-9_]*') >> str
variable_identifier = identifier
data_identifier = identifier
sequential_identifier = identifier
parallel_identifier = identifier

property = identifier
symbol = Token('[^0-9a-zA-Z \t\r\n]')
keyword = Token('[a-z]+')
semi = symbol(';')
comma = symbol(',')
identifier_property = identifier & ~symbol('.') & property > Property

string_single_quote = Token("'(?:\\\\.|[^'\\\\])*'")
string_double_quote = Token('"(?:\\\\.|[^"\\\\])*"')
string = string_single_quote | string_double_quote

# tokens
real_declaration = Token('real') >> (lambda real: str('float')) # c++ only has floats, not reals
integer_declaration = Token('int') >> str
type_ = real_declaration | integer_declaration

real2d_declaration = Token('real2d')
integer2d_declaration = Token('int2d')
data_type = real2d_declaration | integer2d_declaration

real = Token(UnsignedReal()) >> float
integer = Token(UnsignedInteger()) >> int
number = integer | real | keyword('true') >> bool | keyword('false') >> bool

width = integer
height = integer
size = ~symbol('(') & width & ~comma & height & ~symbol(')') > Size._make

#### Expression Parsing ####
# Operator precedence, inside to outside
#  1 parentheses ()
#  2 not, unary minus (!, -)
#  3 multiplication, division (*, /, %)
#  4 addition, subtraction (+, -)
#  5 less than, less than or equal, greater than, greater than or equal (<, <=,>, >=)
#  6 equal, not equal (==, !=)
#  7 and (&&)
#  8 or (||)
#  9 assignment (=)
group2, group3, group4, group5, group6, group7, group8 \
    = Delayed(), Delayed(), Delayed(), Delayed(), Delayed(), Delayed(), Delayed()

expression = Delayed()
primary = Delayed()
parallel_function_call = Delayed()
host_function_call = Delayed()

# first layer, most tightly grouped, is parens and numbers
parens = ~symbol('(') & expression & ~symbol(')')
group1 = parens | number | primary

unary_not = ~symbol('!') & group2 > Not._make
unary_neg = ~symbol('-') & group2 > Neg._make
group2 += unary_not | unary_neg | group1

# third layer, next most tightly grouped, is multiplication
mul = group2 & ~symbol('*') & group3 > Mul._make
div = group2 & ~symbol('/') & group3 > Div._make
mod = group2 & ~symbol('%') & group3 > Mod._make
group3 += mul | div | mod | group2

# fourth layer, least tightly grouped, is addition
add = group3 & ~symbol('+') & group4 > Add._make
sub = group3 & ~symbol('-') & group4 > Sub._make
group4 += add | sub | group3

less_than              = group4 & ~symbol('<')   & group5 > LessThan._make
less_than_or_equal     = group4 & ~symbol('<') & ~symbol('=') & group5 > LessThanOrEqual._make
greater_than           = group4 & ~symbol('>')   & group5 > GreaterThan._make
greather_than_or_equal = group4 & ~symbol('>') & ~symbol('=') & group5 > GreaterThanOrEqual._make
group5 += less_than | less_than_or_equal | greater_than | greather_than_or_equal | group4

equal     = group5 & ~symbol('=')[2] & group6 > Equal._make
not_equal = group5 & ~symbol('!') & ~symbol('=') & group6 > NotEqual._make
group6 += equal | not_equal | group5

boolean_and = group6 & ~symbol('&')[2] & group7 > BooleanAnd
group7 += boolean_and | group6

boolean_or = group7 & ~symbol('|')[2] & group8 > BooleanOr
group8 += boolean_or | group7

assignment = (identifier | identifier_property) & ~symbol('=') & expression > Assignment
expression += assignment | group8

expression_list = expression[0:, ~comma] > Expressions

statement, block = Delayed(), Delayed()
return_ = ~keyword('return') & expression & ~semi > Return
break_ =  ~keyword('break') & ~semi > Break
if_ =     ~keyword('if') & ~symbol('(') & expression & ~symbol(')') & statement & Optional(~keyword('else') & statement) > If
while_ =  ~keyword('while') & ~symbol('(') & expression & ~symbol(')') & statement > While
statement += ~semi | parallel_function_call | host_function_call | ((expression & ~semi) > Statement) | return_ | break_ | if_ | while_ | block

#### Top Level Program Matching ####
parameter_declaration = (type_ | keyword('window')) & identifier > Parameter._make
#parameter_declaration = type_ & identifier > Parameter
parameter_declaration_list = parameter_declaration[0:, ~comma] > Parameters

initialization = ~symbol('=') & expression > Initialization

variable_declaration = type_ & identifier & Optional(initialization) & ~semi > VariableDeclaration
data_declaration = data_type & identifier & size & Optional(initialization) & ~semi > DataDeclaration
sequential_function_declaration = ~Token('sequential') & identifier & ~symbol('(') & Optional(parameter_declaration_list) & ~symbol(')') & block > SequentialFunctionDeclaration
parallel_function_declaration = ~Token('parallel') & identifier & ~symbol('(') & Optional(parameter_declaration_list) & ~symbol(')') & block > ParallelFunctionDeclaration

#Built-in Sequential functions
sequential_print = ~keyword('print') & ~symbol('(') & string & (~comma & expression)[:] & ~symbol(')') > SequentialPrint
generic_sequential_function_call = identifier & ~symbol('(') & expression_list & ~symbol(')') > SequentialFunctionCall

sequential_function_call = sequential_print | generic_sequential_function_call
primary += sequential_function_call | identifier | identifier_property

## Semi-parallel functions
#data_read  = ~symbol(':') & ~keyword('read') & ~symbol('(') & data_identifier & ~symbol(')') & ~semi > Read
#data_write = ~symbol(':') & ~keyword('write') & ~symbol('(') & data_identifier & ~comma & filename & ~symbol(')') & ~semi > Write
data_print = ~symbol(':') & ~keyword('print') & ~symbol('(') & data_identifier & ~symbol(')') & ~semi > Print
host_function_call += data_print

## Parallel functions
parallel_random = data_identifier & ~symbol('=') & \
                  ~symbol(':') & ~keyword('random') & ~symbol('(') & ~symbol(')') & ~semi > ParallelRandom
parallel_reduce = variable_identifier & ~symbol('=') & \
                  ~symbol(':') & ~keyword('reduce') & ~symbol('(') & data_identifier & Optional(~comma & parallel_identifier) & ~symbol(')') & ~semi > ParallelReduce
parallel_sort = ~symbol(':') & ~keyword('sort') & ~symbol('(') & data_identifier & Optional(~comma & parallel_identifier) & ~symbol(')') & ~semi > ParallelSort
genric_parallel_function_call = data_identifier & ~symbol('=') & ~symbol(':') & identifier & ~symbol('(') & expression_list & ~symbol(')') & ~semi > ParallelFunctionCall

parallel_function_call += parallel_random | parallel_reduce | parallel_sort | genric_parallel_function_call

## Now we can define the last bits
block += ~symbol('{') & (statement | variable_declaration)[0:] & ~symbol('}') > Block

declaration_list = (data_declaration | variable_declaration | sequential_function_declaration | parallel_function_declaration | statement)[0:]

program = declaration_list > Program


# Taken from http://www.saltycrane.com/blog/2007/11/remove-c-comments-python/
def remove_multi_line_comments(text):
    """ remove c-style comments.
        text: blob of text with comments (can include newlines)
        returns: text with comments removed
    """
    pattern = r"""
                            ##  --------- COMMENT ---------
           /\*              ##  Start of /* ... */ comment
           [^*]*\*+         ##  Non-* followed by 1-or-more *'s
           (                ##
             [^/*][^*]*\*+  ##
           )*               ##  0-or-more things which don't start with /
                            ##    but do end with '*'
           /                ##  End of /* ... */ comment
         |                  ##  -OR-  various things which aren't comments:
           (                ## 
                            ##  ------ " ... " STRING ------
             "              ##  Start of " ... " string
             (              ##
               \\.          ##  Escaped char
             |              ##  -OR-
               [^"\\]       ##  Non "\ characters
             )*             ##
             "              ##  End of " ... " string
           |                ##  -OR-
                            ##
                            ##  ------ ' ... ' STRING ------
             '              ##  Start of ' ... ' string
             (              ##
               \\.          ##  Escaped char
             |              ##  -OR-
               [^'\\]       ##  Non '\ characters
             )*             ##
             '              ##  End of ' ... ' string
           |                ##  -OR-
                            ##
                            ##  ------ ANYTHING ELSE -------
             .              ##  Anything other char
             [^/"'\\]*      ##  Chars which doesn't start a comment, string
           )                ##    or escape
    """
    regex = re.compile(pattern, re.VERBOSE|re.MULTILINE|re.DOTALL)
    noncomments = [m.group(2) for m in regex.finditer(text) if m.group(2)]

    return "".join(noncomments)

# And then my code for removing single line comments. much more consise!
def remove_single_line_comments(code):
  comment_remover = (~Regexp(r'//[^\n]*') | Any())[:, ...]
  return comment_remover.parse(code)[0]

def parse(code):
  code = remove_multi_line_comments(code)
  code = remove_single_line_comments(code)

  return program.parse(code)[0]

def main():
  import sys
  with open(sys.argv[1], 'r') as f:
    code = ''.join(f.readlines())

  code = remove_multi_line_comments(code)
  code = remove_single_line_comments(code)

  print(program.parse(code)[0])

if __name__ == '__main__':
  main()
