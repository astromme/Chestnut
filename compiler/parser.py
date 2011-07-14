#!/usr/bin/env python

import re
from nodes import *


def with_line(node):
    def wrapper(results, stream_in, stream_out):
        return node([results, s_delta(stream_in)[1], s_delta(stream_out)[1]])
    return wrapper

def set_and_return(obj, **kwargs):
    for name in kwargs:
        obj.__setattr__(name, kwargs[name])

    return obj

base = Token('[a-zA-Z][a-zA-Z0-9_]*')
function = base
identifier = base >> Symbol
variable_identifier = identifier
data_identifier = identifier
sequential_identifier = identifier
parallel_identifier = identifier

property = identifier
symbol = Token('[^0-9a-zA-Z \t\r\n]')
keyword = Token('[a-z]+')
semi = symbol(';')
colon = symbol(':')
comma = symbol(',')
dot = symbol('.')
identifier_property = identifier & ~dot & property > Property

string_single_quote = Token("'(?:\\\\.|[^'\\\\])*'") >> (lambda obj: String(obj[1:-1]))
string_double_quote = Token('"(?:\\\\.|[^"\\\\])*"') >> (lambda obj: String(obj[1:-1]))
string = string_single_quote | string_double_quote

# tokens
int2d_declaration = Token('IntArray2d') >> Type
real2d_declaration = Token('RealArray2d') >> Type
color2d_declaration = Token('ColorArray2d') >> Type
data_type = real2d_declaration | int2d_declaration | color2d_declaration

int_window2d_declaration = Token('IntWindow2d') >> Type
real_window2d_declaration = Token('RealWindow2d') >> Type
color_window2d_declaration = Token('ColorWindow2d') >> Type
window_type = real_window2d_declaration | int_window2d_declaration | color_window2d_declaration

int_declaration = Token('Int') >> Type
real_declaration = Token('Real') >> Type
color_declaration = Token('Color') >> Type
bool_declaration = Token('Bool') >> Type
point2d_declaration = Token('Point2d') >> Type
size2d_declaration = Token('Size2d') >> Type
type_ = real_declaration | int_declaration | color_declaration | bool_declaration | point2d_declaration | size2d_declaration | window_type

real = Token(UnsignedReal()) >> Real
integer = Token(UnsignedInteger()) >> Integer
number = integer | real | keyword('yes') >> Bool | keyword('no') >> Bool | keyword('true') >> Bool | keyword('false') >> Bool

width = integer
height = integer

unopened_size_block = (width & comma & height & symbol(']')) ** make_error('no [ before {out_rest!s}') & symbol(']')
unclosed_size_block = (symbol('[') & width & comma & height) ** make_error('Datablock size specification is missing a closing ]')

size = Or(
        ~symbol('[') & width & ~comma & height & ~symbol(']') > Size,
        unopened_size_block,
        unclosed_size_block
        )

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
group2, group3_product, group4_sum, group5, group6, group7, group8 \
    = Delayed(), Delayed(), Delayed(), Delayed(), Delayed(), Delayed(), Delayed()

expression = Delayed()
primary = Delayed()
parallel_function_call = Delayed()
host_function_call = Delayed()

# first layer, most tightly grouped, is parens and numbers
parens = ~symbol('(') & expression & ~symbol(')')
group1 = parens | number | primary

unary_not = ~symbol('!') & group2 > Not
unary_neg = ~symbol('-') & group2 > Neg
binary_mod = group1 & ~symbol('%') & group2 > Mod
group2 += unary_not | unary_neg | binary_mod | group1

# third layer, next most tightly grouped, is multiplication
multiplier = ~symbol('*') & group2
inverse    = ~symbol('/') & group2 > Inverse
group3_pass_on  = group2
group3_include  = group2 & (multiplier | inverse)[1:] > Product
group3_product += group3_pass_on | group3_include

# fourth layer, less tightly grouped, is addition
addend   = ~symbol('+') & group3_product
negative = ~symbol('-') & group3_product > Negative
group4_pass_on = group3_product
group4_include = group3_product & (addend | negative)[1:] > Sum
group4_sum    += group4_pass_on | group4_include

#group4end = Delayed()
#add = ~symbol('+') & group3 & group4end > List
#sub = ~symbol('-') & group3 & group4end > List
#group4end += Optional(add | sub)
#group4 += group3 & group4end > List


less_than              = group4_sum & ~symbol('<')   & group5 > LessThan
less_than_or_equal     = group4_sum & ~symbol('<') & ~symbol('=') & group5 > LessThanOrEqual
greater_than           = group4_sum & ~symbol('>')   & group5 > GreaterThan
greather_than_or_equal = group4_sum & ~symbol('>') & ~symbol('=') & group5 > GreaterThanOrEqual
group5 += less_than | less_than_or_equal | greater_than | greather_than_or_equal | group4_sum

equal     = group5 & ~symbol('=')[2] & group6 > Equal
not_equal = group5 & ~symbol('!') & ~symbol('=') & group6 > NotEqual
group6 += equal | not_equal | group5

boolean_and = group6 & ~symbol('&')[2] & group7 > BooleanAnd
group7 += boolean_and | group6

boolean_or = group7 & ~symbol('|')[2] & group8 > BooleanOr
group8 += boolean_or | group7

assignment = (identifier | identifier_property) & ~symbol('=') & expression > Assignment
expression += assignment | group8

expression_list = expression[0:, ~comma] > Expressions

statement, block, parallel_context = Delayed(), Delayed(), Delayed()
return_ = ~keyword('return') & expression & ~semi > Return
break_ =  ~keyword('break') & ~semi > Break
if_ =     ~keyword('if') & ~symbol('(') & expression & ~symbol(')') & statement & Optional(~keyword('else') & statement) > If
while_ =  ~keyword('while') & ~symbol('(') & expression & ~symbol(')') & statement > While
#semicolin_statement = Or(
#                        Optional(return_ | break_) & ~semi,
#                        Optional(return_ | break_) 
statement += ~semi | parallel_context | ((expression & ~semi) > Statement) | return_ | break_ | if_ | while_ | block

#### Top Level Program Matching ####
parameter_declaration = (type_ | data_type ) & identifier > Parameter
#parameter_declaration = type_ & identifier > Parameter
parameter_declaration_list = parameter_declaration[0:, ~comma] > Parameters

initialization = ~symbol('=') & expression > VariableInitialization
data_initialization = ~symbol('=') & parallel_function_call > DataInitialization

variable_declaration = type_ & identifier & Optional(initialization | data_initialization) & ~semi > VariableDeclaration
data_declaration = data_type & identifier & size & Optional(data_initialization) & ~semi > DataDeclaration
sequential_function_declaration = ~Token('sequential') & type_ & identifier & ~symbol('(') & Optional(parameter_declaration_list) & ~symbol(')') & block > SequentialFunctionDeclaration
parallel_function_declaration = ~Token('parallel') & type_ & identifier & ~symbol('(') & Optional(parameter_declaration_list) & ~symbol(')') & block > ParallelFunctionDeclaration

#Built-in Sequential functions
sequential_print = ~keyword('print') & ~symbol('(') & string & (~comma & expression)[:] & ~symbol(')') > Print
generic_function_call = identifier & ~symbol('(') & expression_list & ~symbol(')') > FunctionCall

sequential_function_call = sequential_print | generic_function_call
primary += parallel_function_call | host_function_call | sequential_function_call | identifier | identifier_property

## Host Data functions
#data_read  = ~symbol(':') & ~keyword('read') & ~symbol('(') & data_identifier & ~symbol(')') & ~semi > Read
#data_write = ~symbol(':') & ~keyword('write') & ~symbol('(') & data_identifier & ~comma & filename & ~symbol(')') & ~semi > Write
data_print = ~keyword('print') & ~symbol('(') & data_identifier & ~symbol(')') > DataPrint
data_display = ~keyword('display') & ~symbol('(') & data_identifier & Optional(~comma & parallel_identifier) & ~symbol(')') > DataDisplay
host_function_call += data_print | data_display

## Parallel functions
min_value = max_value = integer
parallel_random = ~function('random') & ~symbol('(') & Optional(min_value & ~comma & max_value) & ~symbol(')') > Random

parallel_reduce = ~function('reduce') & ~symbol('(') & data_identifier & Optional(~comma & parallel_identifier) & ~symbol(')') > ParallelReduce

parallel_sort   = ~function('sort') & ~symbol('(') & data_identifier & Optional(~comma & parallel_identifier) & ~symbol(')') > ParallelSort


parallel_function_call += parallel_random | parallel_reduce | parallel_sort

## Now we can define the last bits
unclosed_block = (~symbol('{') & (statement | variable_declaration)[0:]) ** make_error('block is missing a closing }}')

block += Or(
        ~symbol('{') & (statement | variable_declaration)[0:] & ~symbol('}') > Block, # ** with_line(Block)
        unclosed_block
        )

foreach_input_parameter = identifier & ~keyword('in') & identifier > ForeachInputParameter
foreach_output_parameter = identifier & ~symbol('!') & ~keyword('in') & identifier > ForeachOutputParameter
foreach_parameter = foreach_output_parameter | foreach_input_parameter
parallel_context += ~keyword('foreach') & (foreach_parameter[0:, ~comma] > List) & ((variable_declaration | statement)[0:] > List) & ~keyword('end') > ParallelContext

declaration_list = (data_declaration | variable_declaration | sequential_function_declaration | parallel_function_declaration | statement)[0:]

program = (declaration_list > Program) >> sexpr_throw

parser = program.get_parse()


def maybeWhitespace(text, nonWhitespace):
    if nonWhitespace:
        return nonWhitespace
    else:
        return whitespace(text)

def whitespace(comment):
    return ' ' * len(comment)

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
    whitespace_blanked = [maybeWhitespace(m.group(0), m.group(2)) for m in regex.finditer(text)]

    return "".join(whitespace_blanked)

# And then my code for removing single line comments. much more consise!
def remove_single_line_comments(code):
  comment_remover = (Regexp(r'//[^\n]*') >> whitespace | Any())[:, ...]
  return comment_remover.parse(code)[0]

def parse(original_code):
  code = remove_multi_line_comments(original_code)
  code = remove_single_line_comments(code)

  try:
    parse = parser(code)[0]
  except SyntaxError, e:
    e.text = original_code.split('\n')[e.lineno - 1]
    raise e

  return parse

def parse_file(filename):
  with open(filename, 'r') as f:
    original_code = ''.join(f.readlines())

  code = remove_multi_line_comments(original_code)
  code = remove_single_line_comments(code)

  try:
    parse = parser(code)[0]
  except SyntaxError, e:
    e.filename = filename
    e.text = original_code.split('\n')[e.lineno - 1]
    raise e

  return parse


def main():
  import sys

  if len(sys.argv) != 2:
      print('Usage: %s input' % sys.argv[0])
      sys.exit(1)

  with open(sys.argv[1], 'r') as f:
    original_code = ''.join(f.readlines())

  code = remove_multi_line_comments(original_code)
  code = remove_single_line_comments(code)

  try:
    print(parser(code)[0])
  except SyntaxError, e:
    e.filename = sys.argv[1]
    e.text = original_code.split('\n')[e.lineno - 1]
    raise e

if __name__ == '__main__':
  main()
