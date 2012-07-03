#!/usr/bin/env python

import re
from .nodes import *
from .builtins import built_in_functions

from . import symboltable

from lepl import Delayed, Token, Or, And, UnsignedReal, UnsignedInteger, Optional, Regexp, Any
from lepl import s_delta, s_fmt, s_deepest, sexpr_throw, make_error, sexpr_fold
from lepl.stream.maxdepth import FullFirstMatchException


def full_first_match_exception_init(filename):
    def init(self, stream):
        super(FullFirstMatchException, self).__init__(
            s_fmt(s_deepest(stream),
                'Chestnut stumbled at somewhere around {rest} ({location}). Check for syntax errors. (file: ' + str(filename) + ')'))
        self.deepest = s_deepest(stream)
        self.kargs = self.deepest[1].kargs(self.deepest[0])

    return init

# Delayed matchers can be used before they are defined
group2, group3_product, group4_sum, group5, group6, group7, group8 = [Delayed() for _ in range(7)]

variable_declaration = Delayed()

expression = Delayed()
primary = Delayed()
function_call = Delayed()

#Line Helpers
def with_line(node):
    def wrapper(results, stream_in, stream_out, *kargs):
        start_line = s_delta(stream_in)[1]
        try:
            end_line = s_delta(stream_out)[1]
        except StopIteration:
            end_line = 'eof'

        n = node(results)
        n.context.start_line = start_line
        n.context.end_line = end_line

        return n

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

symbol = Token('[^0-9a-zA-Z \t\r\n]')
keyword = Token('[a-z]+')
semi = symbol(';')
colon = symbol(':')
comma = symbol(',')
dot = symbol('.')

property = ~dot & (function_call | identifier)
property_list = (function_call | identifier) & property[1:] > Property
array_reference = ((function_call | identifier | property_list) & ~symbol('[') & expression[1:,~comma] & ~symbol(']')) ** with_line(ArrayReference)

string_single_quote = Token("'(?:\\\\.|[^'\\\\])*'") >> (lambda obj: String(obj[1:-1]))
string_double_quote = Token('"(?:\\\\.|[^"\\\\])*"') >> (lambda obj: String(obj[1:-1]))
string = string_single_quote | string_double_quote


def generate_type_tokens(iterable):
    for type in iterable:
        yield Token(type) >> Type

scalar_type        = Or(*generate_type_tokens(symboltable.scalar_types))
array_type         = Or(*generate_type_tokens(symboltable.array_types))
array_element_type = Or(*generate_type_tokens(symboltable.array_element_types))

type_ = scalar_type | array_type | array_element_type

real = Token(UnsignedReal()) >> Real
integer = Token(UnsignedInteger()) >> Integer
number = integer | real
boolean = keyword('yes') >> Bool | keyword('no') >> Bool | keyword('true') >> Bool | keyword('false') >> Bool

width = integer
height = integer
depth = integer

unopened_size_block = (width & Optional(~comma & height & Optional(~comma & depth)) & symbol(']')) ** make_error('no [ before {out_rest!s}') & symbol(']')
unclosed_size_block = (symbol('[') & width & Optional(~comma & height & Optional(~comma & depth))) ** make_error('Array size specification is missing a closing ]')

size = Or(
        (~symbol('[') & width
                      & Optional(~comma & height
                        & Optional(~comma & depth))
       & ~symbol(']')) ** with_line(Size),

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

# first layer, most tightly grouped, is parens and numbers
parens = ~symbol('(') & expression & ~symbol(')')
group1 = parens | boolean | number | string | primary

unary_not = (~symbol('!') & group2) > Not
unary_neg = (~symbol('-') & group2) > Neg
binary_mod = (group1 & ~symbol('%') & group2) > Mod
group2 += unary_not | unary_neg | binary_mod | group1

# third layer, next most tightly grouped, is multiplication
multiplier = ~symbol('*') & group2
inverse    = ~symbol('/') & group2 > Inverse
group3_pass_on  = group2
group3_include  = (group2 & (multiplier | inverse)[1:]) > Product
group3_product += group3_pass_on | group3_include

# fourth layer, less tightly grouped, is addition
addend   = ~symbol('+') & group3_product
negative = ~symbol('-') & group3_product > Neg
group4_pass_on = group3_product
group4_include = (group3_product & (addend | negative)[1:]) > Sum
group4_sum    += group4_pass_on | group4_include

#group4end = Delayed()
#add = ~symbol('+') & group3 & group4end > List
#sub = ~symbol('-') & group3 & group4end > List
#group4end += Optional(add | sub)
#group4 += group3 & group4end > List


less_than              = (group4_sum & ~symbol('<')   & group5) > LessThan
less_than_or_equal     = (group4_sum & ~symbol('<') & ~symbol('=') & group5) > LessThanOrEqual
greater_than           = (group4_sum & ~symbol('>')   & group5) > GreaterThan
greather_than_or_equal = (group4_sum & ~symbol('>') & ~symbol('=') & group5) > GreaterThanOrEqual
group5 += less_than | less_than_or_equal | greater_than | greather_than_or_equal | group4_sum

equal     = (group5 & ~symbol('=')[2] & group6) > Equal
not_equal = (group5 & ~symbol('!') & ~symbol('=') & group6) > NotEqual
group6 += equal | not_equal | group5

boolean_and = (group6 & ~symbol('&')[2] & group7) > BooleanAnd
group7 += boolean_and | group6

boolean_or = (group7 & ~symbol('|')[2] & group8) > BooleanOr
group8 += boolean_or | group7

assignment   = ((identifier | property_list) & ~symbol('=') & expression) > Assignment
assign_plus  = ((identifier | property_list) & ~symbol('+') & ~symbol('=') & expression) > AssignPlus
assign_minus = ((identifier | property_list) & ~symbol('-') & ~symbol('=') & expression) > AssignMinus
assign_mul   = ((identifier | property_list) & ~symbol('*') & ~symbol('=') & expression) > AssignMul
assign_div   = ((identifier | property_list) & ~symbol('/') & ~symbol('=') & expression) > AssignDiv
expression += assignment | assign_plus | assign_minus | assign_mul | assign_div | group8

expression_list = expression[0:, ~comma] > Expressions

statement, block, parallel_context = Delayed(), Delayed(), Delayed()
return_ = (~keyword('return') & expression & ~semi) ** with_line(Return)
break_ = ( ~keyword('break') & ~semi) ** with_line(Break)
if_ = (    ~keyword('if') & ~symbol('(') & expression & ~symbol(')') & statement & Optional(~keyword('else') & statement)) ** with_line(If)
while_ = ( ~keyword('while') & ~symbol('(') & expression & ~symbol(')') & statement) ** with_line(While)
for_ = ( ~keyword('for') & ~symbol('(') &
                                 (expression & ~symbol(';') | variable_declaration) &
                                 expression &
                                 ~symbol(';') & expression &
                           ~symbol(')') & statement) ** with_line(For)
#semicolin_statement = Or(
#                        Optional(return_ | break_) & ~semi,
#                        Optional(return_ | break_) 
statement += ~semi | parallel_context | ((expression & ~semi) > Statement) | return_ | break_ | if_ | while_ | for_ | block

#### Top Level Program Matching ####
parameter_declaration = ((type_ | array_type ) & identifier) > Parameter
parameter_declaration_list = (parameter_declaration[0:, ~comma]) > Parameters

initialization = (~symbol('=') & expression) ** with_line(VariableInitialization)
data_initialization = (~symbol('=') & expression) ** with_line(ArrayInitialization)

variable_declaration += (type_ & identifier & Optional(initialization) & ~semi) ** with_line(VariableDeclaration)
data_declaration = (array_type & identifier & size & Optional(data_initialization) & ~semi) ** with_line(ArrayDeclaration)
sequential_function_declaration = (~Token('sequential') & (type_ | array_type) & identifier & ~symbol('(') & Optional(parameter_declaration_list) & ~symbol(')') & block) ** with_line(SequentialFunctionDeclaration)
parallel_function_declaration = (~Token('parallel') & type_ & identifier & ~symbol('(') & Optional(parameter_declaration_list) & ~symbol(')') & block) ** with_line(ParallelFunctionDeclaration)
object_declaration = ~keyword('object') & identifier & ~symbol('{') & (((type_ & identifier & ~semi) > List)[0:] > List) & ~symbol('}') > ObjectDeclaration

#Built-in Sequential functions
function_call += (identifier & ~symbol('(') & expression_list & ~symbol(')')) ** with_line(FunctionCall)
primary += function_call | identifier | property_list | array_reference

## Now we can define the last bits
unclosed_block = (~symbol('{') & (statement | variable_declaration | data_declaration)[0:]) ** make_error('block is missing a closing }}')


block += Or(
        (~symbol('{') & (statement | variable_declaration | data_declaration)[0:] & ~symbol('}')) > Block,
        unclosed_block
        )

foreach_parameter = (identifier & ~keyword('in') & identifier) ** with_line(ForeachParameter)
parallel_context += (~keyword('foreach') & (foreach_parameter[0:, ~comma] > List) & ((variable_declaration | statement)[0:] > List) & ~keyword('end')) ** with_line(ParallelContext)

declaration_list = (object_declaration | data_declaration | variable_declaration | sequential_function_declaration | parallel_function_declaration | statement)[0:]


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

# And then my code for removing single line comments. much more concise!
def remove_single_line_comments(code):
  comment_remover = (Regexp(r'//[^\n]*') >> whitespace | Any())[:, ...]
  try:
    return comment_remover.parse(code)[0]
  except IndexError:
    return '' # we ain't gots no code (empty file)


def parse(original_code, from_file='unknown file'):
    FullFirstMatchException.__init__ = full_first_match_exception_init(from_file)

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
