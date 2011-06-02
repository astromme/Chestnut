from lepl import *
from collections import namedtuple, defaultdict
from templates import *
from symboltable import SymbolTable, Scope, Keyword, Variable, Data, ParallelFunction, SequentialFunction

symbolTable = SymbolTable()

#helpers so that we don't get errors about undefined to_cpp methods
class str(str): to_cpp = lambda self, env=None: self
class int(int): to_cpp = lambda self, env=None: str(self)
class float(float): to_cpp = lambda self, env=None: str(self)

#helper to convert sub-members into strings
def cpp_tuple(obj, env=defaultdict(bool)):
    return tuple(map(lambda element: element.to_cpp(env), obj))

#helper to do nice code indenting
def indent(code, indent_first_line=True):
    if indent_first_line:
        code = '  ' + code
    return code.replace('\n', '\n  ')

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
  def to_cpp(self, env=defaultdict(bool)):
    return "!%s" % cpp_tuple(self, env)
class Neg(namedtuple('Neg', ['value'])):
  def to_cpp(self, env=defaultdict(bool)):
    return "-%s" % cpp_tuple(self, env)

class Mul(namedtuple('Mul', ['left', 'right'])):
  def to_cpp(self, env=defaultdict(bool)):
    return "(%s * %s)" % cpp_tuple(self, env)
class Div(namedtuple('Div', ['numerator', 'divisor'])):
  def to_cpp(self, env=defaultdict(bool)):
    return "((float)%s / %s)" % cpp_tuple(self, env)
class Mod(namedtuple('Mod', ['number', 'modder'])):
  def to_cpp(self, env=defaultdict(bool)):
    return "((int)%s %% %s)" % cpp_tuple(self, env)

class Add(namedtuple('Add', ['left', 'right'])):
  def to_cpp(self, env=defaultdict(bool)):
    return "(%s + %s)" % cpp_tuple(self, env)
class Sub(namedtuple('Sub', ['left', 'right'])):
  def to_cpp(self, env=defaultdict(bool)):
    return "(%s - %s)" % cpp_tuple(self, env)

class LessThan(namedtuple('LessThan', ['left', 'right'])):
  def to_cpp(self, env=defaultdict(bool)):
    return "(%s < %s)" % cpp_tuple(self, env)
class LessThanOrEqual(namedtuple('LessThanOrEqual', ['left', 'right'])):
  def to_cpp(self, env=defaultdict(bool)):
    return "(%s <= %s)" % cpp_tuple(self, env)
class GreaterThan(namedtuple('GreaterThan', ['left', 'right'])):
  def to_cpp(self, env=defaultdict(bool)):
    return "(%s > %s)" % cpp_tuple(self, env)
class GreaterThanOrEqual(namedtuple('GreaterThanOrEqual', ['left', 'right'])):
  def to_cpp(self, env=defaultdict(bool)):
    return "(%s >= %s)" % cpp_tuple(self, env)

class Equal(namedtuple('Equal', ['left', 'right'])):
  def to_cpp(self, env=defaultdict(bool)):
    return "(%s == %s)" % cpp_tuple(self, env)
class NotEqual(namedtuple('NotEqual', ['left', 'right'])):
  def to_cpp(self, env=defaultdict(bool)):
    return "(%s != %s)" % cpp_tuple(self, env)

class BooleanAnd(List):
  def to_cpp(self, env=defaultdict(bool)):
    return "(%s && %s)" % cpp_tuple(self, env)

class BooleanOr(List):
  def to_cpp(self, env=defaultdict(bool)):
    return "(%s || %s)" % cpp_tuple(self, env)

class Assignment(List):
  def to_cpp(self, env=defaultdict(bool)):
    return '%s = %s' % cpp_tuple(self, env)
# End Operators

# Other structures
class Program(List): pass

class VariableDeclaration(List):
    def to_cpp(self, env=defaultdict(bool)):
        type, name = self[0], self[1]
        symbolTable.add(Variable(name, type))
        if len(self) == 3: # we have an initialization
            env = defaultdict(bool, variable_to_assign=name)
            return '%s %s;\n%s' % cpp_tuple(self, env)
        else:
            return '%s %s;' % cpp_tuple(self, env)


class DataDeclaration(List):
  def to_cpp(self, env=defaultdict(bool)):
    if len(self) == 2: #only name and type, no size or initialization
      type_, name = self
      symbolTable.add(Data(name, type_, 0, 0)) #TODO: fix to not have 0,0 for uninitialized

    elif len(self) == 3: # adds a size
      type_, name, size = self
      symbolTable.add(Data(name, type_, size.width, size.height))
      return create_data(type_, name, size)

    elif len(self) == 4: # adds an initialization
      type_, name, size, initialization = self
      symbolTable.add(Data(name, type_, size.width, size.height))
      env = defaultdict(bool, data_to_assign=name)
      return '%s\n %s' % (create_data(type_, name, size), initialization.to_cpp(env))

host_function_template = """\
%(location)s%(type)s %(function_name)s(%(parameters)s) %(block)s
"""
class SequentialFunctionDeclaration(List):
    def to_cpp(self, env=defaultdict(bool)):
        env = defaultdict(bool, sequential=True)
        type_, name, parameters, block = self
        symbolTable.add(SequentialFunction(name, type_, parameters, ok_for_device=True))
        symbolTable.createScope()
        for parameter in parameters:
            symbolTable.add(Variable(name=parameter.name, type=parameter.type))

        environment = { 'function_name' : name,
                        'type' : type_,
                        'location' : '__host__ __device__\n',
                        'parameters' : ', '.join(map(lambda param: '%s %s' % param, parameters)),
                        'block' : block.to_cpp(env) }

        host_function = host_function_template % environment

        symbolTable.removeScope()
        return host_function

class ParallelFunctionDeclaration(List):
    def to_cpp(self, env=defaultdict(bool)):
        name, parameters, block = self
        symbolTable.add(ParallelFunction(name, parameters))
        symbolTable.createScope()
        for parameter in parameters:
            symbolTable.add(Variable(name=parameter.name, type=parameter.type))

        device_function = create_device_function(self)

        symbolTable.removeScope()
        return device_function

class Parameters(List): pass
class Block(List):
  def to_cpp(self, env=defaultdict(bool)):
    symbolTable.createScope()
    cpp = '{\n' + indent('\n'.join(cpp_tuple(self, env))) + '\n}'
    symbolTable.removeScope()
    return cpp

class VariableInitialization(List):
  def to_cpp(self, env=defaultdict(bool)):
    return '%s = %s;' % (env['variable_to_assign'], self[0].to_cpp(env))

class DataInitialization(List):
    def to_cpp(self, env=defaultdict(bool)):
        return self[0].to_cpp(env)

class SequentialPrint(List):
    def to_cpp(self, env=defaultdict(bool)):
        num_format_placeholders = self[0].to_cpp(env).count('%s')
        num_args = len(self)-1

        if num_format_placeholders != num_args:
            raise CompilerException("Error, there are %s format specifiers but %s arguments to print()" % \
                    (num_format_placeholders, num_args))
        format_substrings = self[0].to_cpp(env).split('%s')

        code = 'std::cout'
        for substring, placeholder in zip(format_substrings[:-1], cpp_tuple(self[1:], env)):
            code += ' << "%s"' % substring
            code += ' << %s' % placeholder
        code += ' << "%s" << std::endl' % format_substrings[-1]

        return code


sequential_function_call_template = """\
%(function_name)s(%(arguments)s)
"""
class SequentialFunctionCall(List):
    def to_cpp(self, env=defaultdict(bool)):
        function = self[0]
        arguments = self[1:][0]

        check_is_symbol(function)
        function = symbolTable.lookup(function)
        check_type(function, SequentialFunction)

        if not len(arguments) == len(function.parameters):
            print arguments
            raise CompilerException("Error, parallel function ':%s' takes %s parameters but %s were given" % \
                    (function.name, len(function.parameters), len(arguments)))


        return sequential_function_call_template % { 'function_name' : function.name,
                                                     'arguments' : ', '.join(cpp_tuple(arguments, env)) }


class Size(namedtuple('Size', ['width', 'height'])): pass

class Parameter(namedtuple('Parameter', ['type', 'name'])): pass
class Statement(List):
  def to_cpp(self, env=defaultdict(bool)):
    return ''.join(cpp_tuple(self, env)) + ';'
class Expressions(List):
  def to_cpp(self, env=defaultdict(bool)):
    return ''.join(cpp_tuple(self, env))

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
    def to_cpp(self, env=defaultdict(bool)):
        check_is_symbol(self[0])
        symbol = symbolTable.lookup(self[0])
        check_type(symbol, Variable) #TODO: Support Data properties

        if symbol.type == 'window':
            return coordinates[self[1]]
        else:
            print self
            raise Exception

class Return(List):
    def to_cpp(self, env=defaultdict(bool)):
        if env['sequential']:
            return 'return %s;' % self[0].to_cpp(env)
        else:
            return '{ // Returning from device function requires 2 steps\n' + indent('thrust::get<0>(t) = %s;\nreturn;' % self[0].to_cpp(env)) + '\n}'

class Break(List):
  def to_cpp(self, env=defaultdict(bool)):
    return 'break;'
class If(List):
  def to_cpp(self, env=defaultdict(bool)):
    if len(self) == 2: # simple if (condition) {statement}
      return 'if (%s) %s' % cpp_tuple(self, env)
    elif len(self) == 3: # full if (condition) {statement} else {statement}
      return 'if (%s) %s else %s' % cpp_tuple(self, env)
    else:
      raise Exception("Wrong type of if statement with %s length" % len(self))

class While(List):
  def to_cpp(self, env=defaultdict(bool)):
    return 'while (%s) %s' % cpp_tuple(self, env)


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
    def to_cpp(self, env=defaultdict(bool)):
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



class ParallelAssignment(List):
    def to_cpp(self, env=defaultdict(bool)):
        env = defaultdict(bool, data_to_assign=self[0])
        return self[1].to_cpp(env)

random_template = """\
%(data)s.randomize(%(limits)s);
"""
class ParallelRandom(List):
    def to_cpp(self, env=defaultdict(bool)):
        output = env['data_to_assign']
        if not output:
            raise InternalException("The environment '%s' doesn't have a 'data_to_assign' variable set" % env)
        check_is_symbol(output)
        output = symbolTable.lookup(output)
        check_type(output, Data)

        if len(self) == 2:
            limits = ', '.join(cpp_tuple(self[0:2]))
        else:
            limits = ''

        return random_template % { 'data' : output.name,
                                   'limits' : limits,
                                   'type' : type_map[output.type] }


reduce_template = """
// Reducing '%(input_data)s' to the single value '%(output_variable)s'

// To do this we first unpad our 2d array, %(input_data)s. Thrust then performs the hard work,
// and the final output is copied back into the padded array %(output_variable)s.
{
  thrust::device_vector<%(type)s> *unpadded = %(input_data)s.unpadded();
  int length = (%(input_data)s.height-2) * (%(input_data)s.width-2);
  %(output_variable)s = thrust::reduce(unpadded->begin(), unpadded->begin()+length);
}
"""
class ParallelReduce(List):
    def to_cpp(self, env=defaultdict(bool)):
        function = None
        output = env['variable_to_assign']
        if not output:
            raise InternalException("The environment '%s' doesn't have a 'data_to_assign' variable set" % env)

        if len(self) == 1:
            input = self[0]
        elif len(self) == 2:
            input, function = self
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
                                       'output_variable' : output.name,
                                       'type' : type_map[input.type] }

sort_template = """
// Sorting '%(input_data)s' and placing it into '%(output_data)s'

// To do this we first unpad our 2d array, %(input_data)s. Thrust then performs the hard work,
// and the final output is copied back into the padded array %(output_data)s.
{
  thrust::device_vector<%(type)s> *unpadded = %(input_data)s.unpadded();
  int length = (%(input_data)s.height-2) * (%(input_data)s.width-2);
  thrust::sort(unpadded->begin(), unpadded->begin()+length);

  %(output_data)s.loadWithUnpaddedData(*unpadded);
}
"""
class ParallelSort(List):
    def to_cpp(self, env=defaultdict(bool)):
        function = None
        output = env['data_to_assign']
        if not output:
            raise InternalException("The environment '%s' doesn't have a 'data_to_assign' variable set" % env)

        if len(self) == 1:
            input = self[0]
        elif len(self) == 2:
            input, function = self
        else:
            raise InternalException("Wrong list length to parallel sort")

        check_is_symbol(input)
        check_is_symbol(output)

        input = symbolTable.lookup(input)
        output = symbolTable.lookup(output)

        check_type(input, Data)
        check_type(output, Data)

        #TODO: Actually use function
        if function:
            print('Warning: functions in reduce() calls are not implemented yet')
            check_is_symbol(function)
            function = symbolTable.lookup(function)
            check_type(function, SequentialFunction)
            return ''

        else:
            return sort_template % { 'input_data' : input.name,
                                     'output_data' : output.name,
                                     'type' : type_map[input.type] }


map_template = """
{
    %(maybe_wraps)s

    Chestnut<%(type)s>::chestnut2dTupleIterator startIterator
        = Chestnut<%(type)s>::startIterator(%(input_data)s, %(output_data)s);

    Chestnut<%(type)s>::chestnut2dTupleIterator endIterator
        = Chestnut<%(type)s>::endIterator(%(input_data)s, %(output_data)s);


    thrust::for_each(startIterator, endIterator, %(function)s_functor(%(variable_arguments)s));

    if (%(input_data)s == %(output_data)s) {
        %(input_data)s.swap();
    }
}
"""
class ParallelFunctionCall(List):
    def to_cpp(self, env=defaultdict(bool)):
        output = env['data_to_assign']
        if not output:
            raise InternalException("The environment '%s' doesn't have a 'data_to_assign' variable set" % env)

        function = self[0]
        arguments = self[1:][0]

        check_is_symbol(function)
        function = symbolTable.lookup(function)
        check_type(function, ParallelFunction)

        if not len(arguments) == len(function.parameters):
            print arguments
            raise CompilerException("Error, parallel function ':%s' takes %s parameters but %s were given" % \
                    (function.name, len(function.parameters), len(arguments)))

        check_is_symbol(output)
        output = symbolTable.lookup(output)
        check_type(output, Data)

        variable_arguments = []
        data_arguments = []
        for argument, parameter in zip(arguments, function.parameters):
            if not parameter.type == 'window':
                variable_arguments.append(argument.to_cpp(env))
            else:
                data_arguments.append(argument.to_cpp(env))


        #TODO: the iterator creation depends on having one input and one output. Not always true, bah.
        input_expression = data_arguments[0]

        return map_template % { 'input_data' : input_expression,
                                'output_data' : output.name,
                                'maybe_wraps' : '\n'.join(map(lambda data: '%s.maybeWrap();' % data, data_arguments)),
                                'variable_arguments' : ','.join(variable_arguments),
                                'type' : type_map[output.type],
                                'function' : function.name }

