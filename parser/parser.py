#!/usr/bin/env python


import re
from lepl import *
from collections import namedtuple

#helpers so that we don't get errors about undefined to_cpp methods
class str(str): to_cpp = lambda self: self
class int(int): to_cpp = lambda self: str(self)
class float(float): to_cpp = lambda self: str(self)

#helper to convert sub-members into strings
def cpp_tuple(obj):
  return tuple(map(lambda element: element.to_cpp(), obj))

# Operators
class Not(namedtuple('Not', ['value'])):
  def to_cpp(self):
    return "!%s" % cpp_tuple(self)
class Neg(namedtuple('Neg', ['value'])):
  def to_cpp(self):
    return "-%s" % cpp_tuple(self)

class Mul(namedtuple('Mul', ['left', 'right'])):
  def to_cpp(self):
    return "%s * %s" % cpp_tuple(self)
class Div(namedtuple('Div', ['numerator', 'divisor'])):
  def to_cpp(self):
    return "%s / %s" % cpp_tuple(self)

class Add(namedtuple('Add', ['left', 'right'])):
  def to_cpp(self):
    return "%s + %s" % cpp_tuple(self)
class Sub(namedtuple('Sub', ['left', 'right'])):
  def to_cpp(self):
    return "%s - %s" % cpp_tuple(self)

class LessThan(namedtuple('LessThan', ['left', 'right'])):
  def to_cpp(self):
    return "%s < %s" % cpp_tuple(self)
class LessThanOrEqual(namedtuple('LessThanOrEqual', ['left', 'right'])):
  def to_cpp(self):
    return "%s <= %s" % cpp_tuple(self)
class GreaterThan(namedtuple('GreaterThan', ['left', 'right'])):
  def to_cpp(self):
    return "%s > %s" % cpp_tuple(self)
class GreaterThanOrEqual(namedtuple('GreaterThanOrEqual', ['left', 'right'])):
  def to_cpp(self):
    return "%s >= %s" % cpp_tuple(self)

class Equal(namedtuple('Equal', ['left', 'right'])):
  def to_cpp(self):
    return "%s == %s" % cpp_tuple(self)
class NotEqual(namedtuple('NotEqual', ['left', 'right'])):
  def to_cpp(self):
    return "%s != %s" % cpp_tuple(self)

class BooleanAnd(List): pass

class BooleanOr(List): pass

class Assignment(List):
  def to_cpp(self):
    return '%s = %s' % cpp_tuple(self)
# End Operators

class Program(List): pass
class VariableDeclaration(List):
  def to_cpp(self):
    if len(self) == 3: # we have an initialization
      return '%s %s = %s;' % cpp_tuple(self)
    else:
      return '%s %s;' % cpp_tuple(self)

class DataDeclaration(List): pass
class Function(List): pass
class Parameters(List): pass
class Block(List):
  def to_cpp(self):
    return '{\n' + '\n'.join(cpp_tuple(self)) + '\n}'
class Initialization(List):
  def to_cpp(self):
    return ''.join(cpp_tuple(self))

class FunctionCall(List): pass

class Size(List): pass
class Parameter(List): pass
class Statement(List):
  def to_cpp(self):
    return ''.join(cpp_tuple(self)) + ';'
class Expressions(List):
  def to_cpp(self):
    return ''.join(cpp_tuple(self))

coordinates = {
               'topLeft' : 1,
               'topCenter' : 2,
               'topRight' : 3,
               'left' : 4,
               'center' : 5,
               'right' : 6,
               'bottomLeft' : 7,
               'bottom' : 8,
               'bottomRight' : 9
               }

class Property(List):
  def to_cpp(self):
    if self[0] == 'window':
      return "thrust::get<%s>(t)" % coordinates[self[1]]
    else:
      print self
      raise Exception

class Return(List):
  def to_cpp(self):
    return 'thrust::get<0>(t) = %s;\nreturn;' % cpp_tuple(self)
class Break(List):
  def to_cpp(self):
    return 'break;'
class If(List):
  def to_cpp(self):
    if len(self) == 2: # simple if (condition) {statement}
      return 'if (%s) %s' % cpp_tuple(self)
    elif len(self) == 3: # full if (condition) {statement} else {statement}
      return 'if (%s) %s else %s' % cpp_tuple(self)
    else:
      raise Exception("Wrong type of if statement with %s length" % len(self))

class While(List): pass



identifier = Token('[a-zA-Z][a-zA-Z0-9_]*') >> str
property = identifier
symbol = Token('[^0-9a-zA-Z \t\r\n]')
keyword = Token('[a-z]+')
semi = symbol(';')
comma = symbol(',')
identifier_property = identifier & ~symbol('.') & property > Property

# tokens
real_declaration = Token('real') >> str
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
size = ~symbol('(') & width & ~comma & height & ~symbol(')') > Size

#### Expression Parsing ####
# Operator precedence, inside to outside
#  1 parentheses ()
#  2 not, unary minus (!, -)
#  3 multiplication, division (*, /)
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

# first layer, most tightly grouped, is parens and numbers
parens = ~symbol('(') & expression & ~symbol(')')
group1 = parens | number | primary 

unary_not = ~symbol('!') & group2 > Not._make
unary_neg = ~symbol('-') & group2 > Neg._make
group2 += unary_not | unary_neg | group1

# third layer, next most tightly grouped, is multiplication
mul = group2 & ~symbol('*') & group3 > Mul._make
div = group2 & ~symbol('/') & group3 > Div._make
group3 += mul | div | group2

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

expression_list = expression[0:, ~comma, ...] > Expressions

statement, block = Delayed(), Delayed()
return_ = ~keyword('return') & expression & ~semi > Return
break_ =  ~keyword('break') & ~semi > Break
if_ =     ~keyword('if') & ~symbol('(') & expression & ~symbol(')') & statement & Optional(~keyword('else') & statement) > If
while_ =  ~keyword('while') & ~symbol('(') & expression & ~symbol(')') & statement > While
statement += ~semi | ((expression & ~semi) > Statement) | return_ | break_ | if_ | while_ | block

#### Top Level Program Matching ####
parameter_declaration = identifier > Parameter
#parameter_declaration = type_ & identifier > Parameter
parameter_declaration_list = parameter_declaration[0:, ~comma] > Parameters

initialization = ~symbol('=') & expression > Initialization

variable_declaration = type_ & identifier & Optional(initialization) & ~semi > VariableDeclaration
data_declaration = data_type & identifier & Optional(size) & Optional(initialization) & ~semi > DataDeclaration
function_declaration = ~Token('function') & identifier & ~symbol('(') & Optional(parameter_declaration_list) & ~symbol(')') & block > Function

function_call = identifier & ~symbol('(') & expression_list & ~symbol(')') > FunctionCall
primary += function_call | identifier | identifier_property

block += ~symbol('{') & (statement | variable_declaration)[0:] & ~symbol('}') > Block

declaration_list = (data_declaration | variable_declaration | function_declaration | statement)[0:]

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
