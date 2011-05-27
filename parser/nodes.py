from lepl import *
from collections import namedtuple
from templates import *
from symboltable import SymbolTable, Scope, Keyword, Variable, Data, ParallelFunction, SequentialFunction

symbolTable = SymbolTable()

#helpers so that we don't get errors about undefined to_cpp methods
class str(str): to_cpp = lambda self: self
class int(int): to_cpp = lambda self: str(self)
class float(float): to_cpp = lambda self: str(self)

#helper to convert sub-members into strings
def cpp_tuple(obj):
  return tuple(map(lambda element: element.to_cpp(), obj))

#helper functions to check for errors and to print them in a nice way
class CompilerException(Exception): pass
class InternalException(Exception): pass

def check_is_symbol(name):
    if not symbolTable.lookup(name):
        raise CompilerException("Error, the symbol '%s' was used but it hasn't been declared yet" % name)

def check_type(symbol, requiredType):
    if type(symbol) is not requiredType:
        raise CompilerException("Error, the symbol '%s' is a %s, but a symbol of type %s was expected" % \
                (symbol.name, type(symbol), requiredType))

def check_dimensions_are_equal(leftData, rightData):
    assert type(leftData) == type(rightData) == Data
    if (not leftData.width == rightData.width) or (not leftData.height == rightData.height):
        raise CompilerException("Error, '%s' (%s, %s) and '%s' (%s, %s) have different dimensions." % \
         (rightData.name, rightData.width, rightData.height, leftData.name, leftData.width, leftData.height))

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
    return "(float)%s / %s" % cpp_tuple(self)
class Mod(namedtuple('Mod', ['number', 'modder'])):
  def to_cpp(self):
    return "(int)%s %% %s" % cpp_tuple(self)

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

class BooleanAnd(List):
  def to_cpp(self):
    return "%s && %s" % cpp_tuple(self)

class BooleanOr(List):
  def to_cpp(self):
    return "%s || %s" % cpp_tuple(self)

class Assignment(List):
  def to_cpp(self):
    return '%s = %s' % cpp_tuple(self)
# End Operators

# Other structures
class Program(List): pass

class VariableDeclaration(List):
  def to_cpp(self):
    type, name = self[0], self[1]
    symbolTable.add(Variable(name, type))
    if len(self) == 3: # we have an initialization
      return '%s %s = %s;' % cpp_tuple(self)
    else:
      return '%s %s;' % cpp_tuple(self)


class DataDeclaration(List):
  def to_cpp(self):
    if len(self) == 2: #only name and type, no size or initialization
      type_, name = self
      symbolTable.add(Data(name, type_, 0, 0)) #TODO: fix to not have 0,0 for uninitialized
      
    elif len(self) == 3: # adds a size
      type_, name, size = self
      symbolTable.add(Data(name, type_, size.width, size.height))
      return create_data(type_, name, size)
    
    elif len(self) == 4: # adds an initialization
      type_, name, size, initialization = self
      symbolTable.add(Data(name, type, size.width, size.height))
      return 'thrust::device_vector<%(type)s> %(name)s(%(size)s) = %(init)s;' % { 'type' : type_map[type_],
                                                                                  'name' : name,
                                                                                  'size' : size.width*size.height,
                                                                                  'init' : initialization.to_cpp() }


class SequentialFunctionDeclaration(List): pass
class ParallelFunctionDeclaration(List):
    def to_cpp(self):
        name, parameters, block = self
        symbolTable.add(ParallelFunction(name, map(lambda param: param[0], parameters)))
        return create_device_function(self)

class Parameters(List): pass
class Block(List):
  def to_cpp(self):
    symbolTable.createScope()
    cpp = '{\n' + '\n'.join(cpp_tuple(self)) + '\n}'
    symbolTable.removeScope()
    return cpp

class Initialization(List):
  def to_cpp(self):
    return ''.join(cpp_tuple(self))

class SequentialPrint(List):
    def to_cpp(self):
        num_format_placeholders = self[0].to_cpp().count('%s')
        num_args = len(self)-1

        if num_format_placeholders != num_args:
            raise CompilerException("Error, there are %s format specifiers but %s arguments to print()" % \
                    (num_format_placeholders, num_args))
        format_substrings = self[0].to_cpp().split('%s')

        code = 'std::cout'
        for substring, placeholder in zip(format_substrings[:-1], cpp_tuple(self[1:])):
            code += ' << "%s"' % substring
            code += ' << %s' % placeholder
        code += ' << "%s" << std::endl' % format_substrings[-1]

        return code


class SequentialFunctionCall(List): pass

class Size(namedtuple('Size', ['width', 'height'])): pass

class Parameter(List): pass
class Statement(List):
  def to_cpp(self):
    return ''.join(cpp_tuple(self)) + ';'
class Expressions(List):
  def to_cpp(self):
    return ''.join(cpp_tuple(self))

# int index = thrust::get<2>(t);
# int paddedWidth = thrust::get<3>(t);
# int paddedHeight = thrust::get<4>(t);
# 
# int x = index % paddedWidth - 1; // -1 for padding
# int y = index / paddedWidth - 1;
# 
# int width = paddedWidth - 2;
# int height = paddedHeight - 2;

coordinates = {
               'topLeft' : r'thrust::get<0>(thrust::get<1>(t))',
               'top' : r'thrust::get<1>(thrust::get<1>(t))',
               'topRight' : r'thrust::get<2>(thrust::get<1>(t))',

               'left' : r'thrust::get<3>(thrust::get<1>(t))',
               'center' : r'thrust::get<4>(thrust::get<1>(t))',
               'right' : r'thrust::get<5>(thrust::get<1>(t))',

               'bottomLeft' : r'thrust::get<6>(thrust::get<1>(t))',
               'bottom' : r'thrust::get<7>(thrust::get<1>(t))',
               'bottomRight' : r'thrust::get<8>(thrust::get<1>(t))',

               'x' : r'(thrust::get<2>(t) % thrust::get<3>(t) - 1)',
               'y' : r'(thrust::get<2>(t) / thrust::get<3>(t) - 1)',
               'width' : r'(thrust::get<3>(t) - 2)',
               'height' : r'(thrust::get<4>(t) - 2)',
               }

class Property(List):
  def to_cpp(self):
    if self[0] == 'window':
      return coordinates[self[1]]
    else:
      print self
      raise Exception

class Return(List):
  def to_cpp(self):
    return '{ thrust::get<0>(t) = %s;\nreturn; }' % cpp_tuple(self)
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

class While(List):
  def to_cpp(self):
    return 'while (%s) %s' % cpp_tuple(self)


class Read(List): pass
class Write(List): pass

print_template = """
{
  // Create host vector to hold data
  thrust::host_vector<%(type)s> hostData(%(length)s);

  // transfer data back to host
  hostData = *%(data)s.mainData;

  printArray2D(hostData, %(width)s, %(height)s, %(padding)s);
}
"""
class Print(List):
    def to_cpp(self):
        data = self[0]
        check_is_symbol(data)
        data = symbolTable.lookup(data)
        check_type(data, Data)
        return print_template % { 'data' : data.name,
                                  'type' : type_map[data.type],
                                  'width' : data.width,
                                  'height' : data.height,
                                  'padding' : 1,
                                  'length' : pad(data.width)*pad(data.height) }




random_template = """
{
  thrust::host_vector<%(type)s> randoms(%(data)s.width*%(data)s.height);
  thrust::generate(randoms.begin(), randoms.end(), rand);
  *%(data)s.mainData = randoms;
}
"""
class ParallelRandom(List):
    def to_cpp(self):
        input = self[0]
        check_is_symbol(input)
        input = symbolTable.lookup(input)
        check_type(input, Data)

        return random_template % { 'data' : input.name,
                                   'type' : type_map[input.type] }

map_template = """
{
    %(input_data)s.maybeWrap();
    Chestnut<%(type)s>::chestnut2dTupleIterator startIterator
        = Chestnut<%(type)s>::startIterator(%(input_data)s, %(output_data)s);

    Chestnut<%(type)s>::chestnut2dTupleIterator endIterator
        = Chestnut<%(type)s>::endIterator(%(input_data)s, %(output_data)s);


    thrust::for_each(startIterator, endIterator, %(function)s_functor());

    if (%(input_data)s == %(output_data)s) {
        //std::cout << "%(input_data)s == %(output_data)s, swapping" << std::endl;
        %(input_data)s.swap();
    }
}
"""
class ParallelMap(List):
  def to_cpp(self):
    output, input, function = self
    check_is_symbol(input)
    check_is_symbol(output)
    check_is_symbol(function)

    input = symbolTable.lookup(input)
    output = symbolTable.lookup(output)
    function = symbolTable.lookup(function)

    check_type(input, Data)
    check_type(output, Data)
    check_type(function, ParallelFunction)
    check_dimensions_are_equal(input, output)

    return map_template % { 'input_data' : input.name,
                            'output_data' : output.name,
                            'type' : type_map[input.type],
                            'function' : function.name }

reduce_template = """
{
  %(output_variable)s = thrust::reduce(%(input_data)s.mainData->begin(), %(input_data)s.mainData->end());
}
"""
class ParallelReduce(List):
    def to_cpp(self):
        function = None
        if len(self) == 2:
            output, input = self
        elif len(self) == 3:
            output, input, function = self
        else:
            raise InternalException("Wrong list length to parallel reduce")

        check_is_symbol(input)
        check_is_symbol(output)

        input = symbolTable.lookup(input)
        output = symbolTable.lookup(output)

        check_type(input, Data)
        check_type(output, Variable)

        #TODO: Actually use function
        if function:
            print('Warning: functions in reduce() calls are not implemented yet')
            check_is_symbol(function)
            function = symbolTable.lookup(function)
            check_type(function, SequentialFunction)
            return ''

        else:
            return reduce_template % { 'input_data' : input.name,
                                       'output_variable' : output.name }

sort_template = """
{
  thrust::reduce(%(data)s.mainData->begin(), %(data)s.mainData->end());
}
"""
class ParallelSort(List):
    def to_cpp(self):
        function = None
        if len(self) == 1:
            input = self[0]
        elif len(self) == 2:
            input, function = self
        else:
            raise InternalException("Wrong list length to parallel sort")

        check_is_symbol(input)
        input = symbolTable.lookup(input)
        check_type(input, Data)

        #TODO: Actually use function
        if function:
            print('Warning: functions in reduce() calls are not implemented yet')
            check_is_symbol(function)
            function = symbolTable.lookup(function)
            check_type(function, SequentialFunction)
            return ''

        else:
            return sort_template % { 'data' : input.name }
