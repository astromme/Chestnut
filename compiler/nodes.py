from lepl import *
from collections import namedtuple, defaultdict
from templates import *
from symboltable import scalar_types, data_types
from symboltable import SymbolTable, Scope, Keyword, StreamVariable, Variable, Window, Data, ParallelFunction, SequentialFunction, DisplayWindow
import random, numpy
from pycuda.gpuarray import GPUArray as DeviceArray
import codepy.cgen as c
import operator as op

import pycuda.curandom
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel

symbolTable = SymbolTable()
global current_name_number
current_name_number = 0

def list_contains(variable, deep_list):
    if len(deep_list) == 0:
        return False
    if deep_list[0] == variable:
        return True

    if isinstance(deep_list[0], list):
        in_subtree = list_contains(variable, deep_list[0])
    else:
        in_subtree = False

    return in_subtree or list_contains(variable, deep_list[1:])

def list_contains_type(deep_list, variable_type):
    for element in deep_list:
        if type(element) == variable_type:
            return True
        elif isinstance(element, list):
            if list_contains_type(element, variable_type):
                return True

    return False

def collect_elements_from_list(deep_list, element_type):
    collection = []
    for element in deep_list:
        if type(element) == element_type:
            collection.append(element)
        elif isinstance(element, List):
            collection.extend(collect_elements_from_list(element, element_type))

    return collection

def next_open_name():
    global current_name_number
    current_name_number += 1
    return str(current_name_number)

#class cuda_context:
#    def __enter__(self):
#        import pycuda.driver
#        from pycuda.tools import make_default_context
#
#        pycuda.driver.init()
#
#        self.context = make_default_context()
#        self.device = self.context.get_device()
#        return self.context
#
#    def __exit__(self, type, value, traceback):
#        print "context pop"
#        self.context.pop()
#        print "context nulling"
#        del self.context
#
#        from pycuda.tools import clear_context_caches
#        print "cache clearing"
#        clear_context_caches()
#        print "done"



def check_is_symbol(name, environment=symbolTable):
    if not environment.lookup(name):
        raise CompilerException("Error, the symbol '%s' was used but it hasn't been declared yet" % name)

def check_type(symbol, *requiredTypes):
    if type(symbol) not in requiredTypes:
        if len(requiredTypes) == 1:
            raise CompilerException("Error, the symbol '%s' is a %s, but a symbol of type %s was expected" % \
                    (symbol.name, type(symbol), requiredTypes[0]))
        else:
            raise CompilerException("Error, the symbol '%s' is a %s, but one of %s was expected" % \
                    (symbol.name, type(symbol), ','.join(requiredTypes)))


def check_dimensions_are_equal(leftData, rightData):
    assert type(leftData) == type(rightData) == Data
    if (not leftData.width == rightData.width) or (not leftData.height == rightData.height):
        raise CompilerException("Error, '%s' (%s, %s) and '%s' (%s, %s) have different dimensions." % \
         (rightData.name, rightData.width, rightData.height, leftData.name, leftData.width, leftData.height))


def extract_line_info(function):
    def wrapper(obj, env=defaultdict(bool)):
        node_info = obj # obj[0:-2]
        start_line, end_line = (0, 0) #obj[-2:]
        return function(obj, node_info, start_line, end_line, env)
    return wrapper



class Type(str):
    def to_cpp(self, env=None):
        return type_map[self]
    def evaluate(self, env):
        return

#helpers so that we don't get errors about undefined to_cpp methods
class Symbol(str):
    def to_cpp(self, env=None):
        check_is_symbol(self)
        return self
    def evaluate(self, env):
        symbol = env[self]
        if type(symbol) == Variable:
            return symbol.value
        elif type(symbol) == Data:
            return symbol
        else:
            raise Exception

class String(str):
    def to_cpp(self, env=None):
        return self
    def evaluate(self, env):
        return self
class Integer(int):
    to_cpp = lambda self, env=None: str(self)
    evaluate = lambda self, env: self
class Real(float):
    to_cpp = lambda self, env=None: str(self)
    evaluate = lambda self, env: self
class Bool(object):
    def __init__(self, value):
        self._value = bool(value == 'yes' or value == 'true')
    def __nonzero__(self):
        return self._value
    def to_cpp(self, env=None):
        return str(self._value).lowercase()
    def evaluate(self, env):
        return self._value

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

#helper functions to emulate return, break
class InterpreterException(Exception): pass
class InterpreterReturn(Exception): pass
class InterpreterBreak(Exception): pass

class RuntimeWindow:
    def __init__(self, x, y, data):
        self.data = data
        self.x = x
        self.y = y

    def prop(self, prop):
        properties = { 'topLeft' : self.topLeft,
                    'top' : self.top,
                    'topRight' : self.topRight,
                    'left' : self.left,
                    'center' : self.center,
                    'bottomLeft' : self.bottomLeft,
                    'bottom' : self.bottom,
                    'bottomRight' : self.bottomRight,
                    'x' : self.x,
                    'y' : self.y,
                    'width' : self.width,
                    'height' : self.height }

        return properties[prop]



    @property
    def width(self):
        return self.data.width
    @property
    def height(self):
        return self.data.height

    @property
    def topLeft(self):
        return self.data.at((self.x-1) % self.width, (self.y-1) % self.height)

    @property
    def top(self):
        return self.data.at((self.x) % self.width, (self.y-1) % self.height)

    @property
    def topRight(self):
        return self.data.at((self.x+1) % self.width, (self.y-1) % self.height)

    @property
    def left(self):
        return self.data.at((self.x-1) % self.width, (self.y) % self.height)

    @property
    def center(self):
        return self.data.at((self.x) % self.width, (self.y) % self.height)

    @property
    def right(self):
        return self.data.at((self.x+1) % self.width, (self.y) % self.height)

    @property
    def bottomLeft(self):
        return self.data.at((self.x-1) % self.width, (self.y+1) % self.height)

    @property
    def bottom(self):
        return self.data.at((self.x) % self.width, (self.y+1) % self.height)

    @property
    def bottomRight(self):
        return self.data.at((self.x+1) % self.width, (self.y+1) % self.height)





### Nodes ###
# Operators
class Not(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "!%s" % cpp_tuple(self, env)
    def evaluate(self, env):
        return not self[0].evaluate(env)
class Neg(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "-%s" % cpp_tuple(self, env)
    def evaluate(self, env):
        return -self[0].evaluate(env)


### Products ###
class Product(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s)" % '*'.join(cpp_tuple(self, env))
    def evaluate(self, env):
        return reduce(op.mul, map(lambda obj: obj.evaluate(env), self))
class Inverse(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(1.0/%s)" % self[0].to_cpp(env)
    def evaluate(self, env):
        return op.truediv(1, self[0].evaluate(env))


### Sums ###
class Sum(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s)" % '+'.join(cpp_tuple(self, env))
    def evaluate(self, env):
        return reduce(op.add, map(lambda obj: obj.evaluate(env), self))

class Negative(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(-%s)" % self[0].to_cpp(env)
    def evaluate(self, env):
        return op.neg(self[0].evaluate(env))

### Old Operators ###
class Mul(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s * %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return self[0].evaluate(env) * self[1].evaluate(env)
class Div(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "((float)%s / %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return self[0].evaluate(env) / self[1].evaluate(env)
class Mod(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "((int)%s %% %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return self[0].evaluate(env) % self[1].evaluate(env)

class Add(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s + %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return self[0].evaluate(env) + self[1].evaluate(env)
class Sub(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s - %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return self[0].evaluate(env) - self[1].evaluate(env)

class LessThan(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s < %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return self[0].evaluate(env) < self[1].evaluate(env)
class LessThanOrEqual(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s <= %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return self[0].evaluate(env) <= self[1].evaluate(env)
class GreaterThan(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s > %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return self[0].evaluate(env) > self[1].evaluate(env)
class GreaterThanOrEqual(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s >= %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return self[0].evaluate(env) >= self[1].evaluate(env)

class Equal(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s == %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return self[0].evaluate(env) == self[1].evaluate(env)
class NotEqual(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s != %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return not (self[0].evaluate(env) == self[1].evaluate(env))

class BooleanAnd(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s && %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return self[0].evaluate(env) and self[1].evaluate(env)

class BooleanOr(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s || %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return self[0].evaluate(env) or self[1].evaluate(env)

class Assignment(List):
    def to_cpp(self, env=defaultdict(bool)):
        output, expression = self
        output = symbolTable.lookup(output)

        return '%s = %s' % (output.cpp_name, expression.to_cpp())
    def evaluate(self, env):
        symbol = env[self[0]]
        symbol.value = self[1].evaluate(env)
        return symbol.value
# End Operators

# Other structures
class Program(List):
    def evaluate(self, env):
        import pycuda.tools

        # effing stupid pycuda context crashes with abort() on deletion
        # so we can't use the nice with blah() as blah: syntax
        #with cuda_context() as context:
        #env['@context'] = context
        device_pool = env['@device_memory_pool'] = pycuda.tools.DeviceMemoryPool()
        host_pool = env['@host_memory_pool'] = pycuda.tools.PageLockedMemoryPool()

        for node in self:
            node.evaluate(env)

class VariableDeclaration(List):
    def to_cpp(self, env=defaultdict(bool)):
        type, name = self[0], self[1]
        symbolTable.add(Variable(name, type))
        if len(self) == 3: # we have an initialization
            env = defaultdict(bool, variable_to_assign=name)
            return '%s %s;\n%s' % cpp_tuple(self, env)
        else:
            return '%s %s;' % cpp_tuple(self, env)
    def evaluate(self, env):
        type, name = self[0:2]
        var = env[name] = Variable(name, type)

        if len(self) == 3: # we have an initialization
            var.value = self[2].evaluate(env)


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

  def evaluate(self, env):
    if len(self) == 2: #only name and type, no size or initialization
      type_, name = self
      env[name] = DeviceArray((0, 0, 0), dtype=numpy_type_map[type_], allocator=dev_pool.allocate)

    elif len(self) == 3: # adds a size
      type_, name, size = self
      (size.width, size.height)
      numpy_type_map[type_]
      env['@device_memory_pool'].allocate
      env[name] = DeviceArray((size.width, size.height), dtype=numpy_type_map[type_], allocator=env['@device_memory_pool'].allocate)

    elif len(self) == 4: # adds an initialization
      type_, name, size, initialization = self
      env[name] = DeviceArray((size.width, size.height), dtype=numpy_type_map[type_], allocator=env['@device_memory_pool'].allocate)
      env['@output'] = env[name]
      symbol = initialization.evaluate(env)

host_function_template = """\
%(location)s%(type)s %(function_name)s(%(parameters)s) %(block)s
"""
class SequentialFunctionDeclaration(List):
    def to_cpp(self, env=defaultdict(bool)):
        env = defaultdict(bool, sequential=True)
        type_, name, parameters, block = self
        symbolTable.add(SequentialFunction(name, type_, parameters, ok_for_device=True, node=self))
        symbolTable.createScope()
        for parameter in parameters:
            symbolTable.add(Variable(name=parameter.name, type=parameter.type))


        environment = { 'function_name' : name,
                        'type' : type_,
                        'location' : '__host__ __device__\n',
                        'parameters' : ', '.join(map(lambda param: '%s %s' % param, tuple(parameters))),
                        'block' : block.to_cpp(env) }

        host_function = host_function_template % environment

        symbolTable.removeScope()
        return host_function

    def evaluate(self, env):
        type_, name, parameters, block = self
        env[name] = SequentialFunction(name, type_, parameters, ok_for_device=True, node=self)

    def run(self, arguments, env):
        type_, name, parameters, block = self

        function_scope = Scope(env)

        for parameter, argument in zip(parameters, arguments):
            symbol = function_scope.add(Variable(name=parameter.name, type=parameter.type))
            symbol.value = argument # argument was evaluated before the function call 

        result = block.evaluate(function_scope)

        return result

class ParallelFunctionDeclaration(List):
    def to_cpp(self, env=defaultdict(bool)):
        #todo: make sure no parallel contexts live inside of here
        type_, name, parameters, block = self
        symbolTable.add(ParallelFunction(name, type_, parameters, node=self))
        symbolTable.createScope()

        windowCount = 0
        for parameter in parameters:
            if parameter.type in data_types:
                symbolTable.add(Window(name=parameter.name, number=windowCount))
                windowCount += 1
            else:
                symbolTable.add(Variable(name=parameter.name, type=parameter.type))

        device_function = create_device_function(self)

        symbolTable.removeScope()
        return device_function

    def evaluate(self, env):
        type_, name, parameters, block = self
        env[name] = ParallelFunction(name, type_, parameters, node=self)

    def run(self, arguments, env):
        name, parameters, block = self

        # TODO: Deal with parallel functions and scope. Minimal env with no global stuff?
        function_scope = Scope()
        env.createScope()

        for parameter, argument in zip(parameters, arguments):
            symbol = env[name] = Variable(name=parameter.name, type=parameter.type)
            symbol.value = argument # argument was evaluated before the function call 

        result = block.evaluate(env)

        env.removeScope()
        return result

class Parameters(List): pass
class Block(List):
    @extract_line_info
    def to_cpp(self, block, start_line, end_line, env):
        print 'start: %s, end: %s' % (start_line, end_line)
        symbolTable.createScope()
        cpp = '{\n' + indent('\n'.join(cpp_tuple(block, env))) + '\n}'
        symbolTable.removeScope()
        return cpp
    @extract_line_info
    def evaluate(self, block, start_line, end_line, env):
        block_scope = Scope(env)
        for statement in self:
            statement.evaluate(block_scope)


class VariableInitialization(List):
    def to_cpp(self, env=defaultdict(bool)):
        return '%s = %s;' % (env['variable_to_assign'], self[0].to_cpp(env))
    def evaluate(self, env):
        return self[0].evaluate(env)

class DataInitialization(List):
    def to_cpp(self, env=defaultdict(bool)):
        return self[0].to_cpp(env)
    def evaluate(self, env):
        return self[0].evaluate(env)


display_template = """\
{
  FunctionIterator _start = makeStartIterator(%(input)s.width, %(input)s.height);

  thrust::copy(thrust::make_transform_iterator(_start, %(display_function)s_functor<%(template_types)s>(%(input)s)),
               thrust::make_transform_iterator(_start+%(input)s.length(), %(display_function)s_functor<%(template_types)s>(%(input)s)),
               _%(input)s_display.displayData());
}
_%(input)s_display.updateGL();
_app.processEvents();
"""
class DataDisplay(List):
    def to_cpp(self, env=defaultdict(bool)):
        data = symbolTable.lookup(self[0])

        print self
        if len(self) == 2:
            display_function = symbolTable.lookup(self[1])
            check_type(display_function, ParallelFunction)
            if display_function.type != 'Color':
                raise CompilerException("Display function '%s' returns data of type '%s'. Display functions must return 'Color'."
                        % (display_function.name, display_function.type))
            parameters = display_function.parameters
            if len(parameters) != 1:
                raise CompilerException("Display function '%s' takes %d parameters. Display functions must only take 1 parameter"
                        % (display_function.name, len(parameters)))
            #if data.type != parameters[0].type:
            #    raise CompilerException("Display function takes a parameter of type '%s' but the data '%s' is of the type '%s'"
            #            % (parameters[0].type, data.name, data.type))

            display_function = display_function.name
        else:
            display_function = '_chestnut_default_color_conversion'

        template_types = 'color, %s' % type_map[data.type]

        display_env = { 'input' : data.name,
                        'template_types' : template_types,
                        'display_function' : display_function }

        code = ""
        if not data.has_display:
            data.has_display = True
            symbolTable.displayWindows.append(DisplayWindow(data.name, 'Chestnut Output [%s]' % data.name, data.width, data.height))
        code += display_template % display_env
        return code

    def evaluate(self, env):
        pass

print_template = """
{
  // Create host vector to hold data
  thrust::host_vector<%(type)s> hostData(%(length)s);

  // transfer data back to host
  %(data)s.copyTo(hostData);

  printArray2D(hostData, %(width)s, %(height)s);
}
"""
class DataPrint(List):
    def to_cpp(self, env=defaultdict(bool)):
        data = self[0]
        check_is_symbol(data)
        data = symbolTable.lookup(data)
        check_type(data, Data)
        return print_template % { 'data' : data.name,
                                  'type' : type_map[data.type],
                                  'width' : data.width,
                                  'height' : data.height,
                                  'length' : data.width*data.height }
    def evaluate(self, env):
        print env[self[0]]

class Print(List):
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
    def evaluate(self, env):
        text = self[0]
        args = map(lambda arg: arg.evaluate(env), self[1:])

        print(text % tuple(args))



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
            raise CompilerException("Error, sequential function ':%s' takes %s parameters but %s were given" % \
                    (function.name, len(function.parameters), len(arguments)))


        return sequential_function_call_template % { 'function_name' : function.name,
                                                     'arguments' : ', '.join(cpp_tuple(arguments, env)) }

    def evaluate(self, env):
        function = self[0]
        arguments = map(lambda obj: obj.evaluate(env), self[1:][0])

        function = env[function]

        try:
            function.node.run(arguments, env)
        except InterpreterBreak:
            raise InterpreterException('Error: caught break statement outside of a loop in function %s', function.name)
        except InterpreterReturn as return_value:
            return return_value[0]


class Size(List):
    @property
    def width(self): return self[0]
    @property
    def height(self): return self[1]

class Parameter(List):
    @property
    def type(self): return self[0]
    @property
    def name(self): return self[1]

class Statement(List):
    def to_cpp(self, env=defaultdict(bool)):
        return ''.join(cpp_tuple(self, env)) + ';'
    def evaluate(self, env):
        return self[0].evaluate(env)

class Expressions(List):
    def to_cpp(self, env=defaultdict(bool)):
        return ''.join(cpp_tuple(self, env))



coordinates = {
        'x' : '_x',
        'y' : '_y',
        'width' : '_width',
        'height' : '_height' }

color_properties = { # Screen is apparently BGR not RGB
        'red' : 'z',
        'green' : 'y',
        'blue' : 'x',
        'alpha' : 'w' }


property_template = """\
%(name)s.%(property)s(%(parameters)s)\
"""
class Property(List):
    def to_cpp(self, env=defaultdict(bool)):
        name, property_ = self
        check_is_symbol(name)
        symbol = symbolTable.lookup(name)


        if type(symbol) == Window:
            if property_ not in ['topLeft', 'top', 'topRight', 'left', 'center', 'right', 'bottomLeft', 'bottom', 'bottomRight']:
                raise CompilerException("Error, property '%s' not part of the data window '%s'" % (property_, name))
            return property_template % { 'name' : name,
                                         'property' : property_,
                                         'parameters' : coordinates['x'] + ', ' + coordinates['y'] }
        elif type(symbol) == Keyword and symbol.name == 'parallel':
            if property_ in coordinates:
                return coordinates[property_];
            else:
                raise CompilerException('Property %s not found for function %s' % (property_, name))
        elif type(symbol) == Variable:
            if symbol.type == 'Color':
                return '%s.%s' % (symbol.name, color_properties[property_])
            else:
                raise Exception('unknown property variable type')
        else:
            print symbolTable
            print self
            raise Exception('unknown property type')

    def evaluate(self, env):
        check_is_symbol(self[0], env)
        symbol = env[self[0]]
        #check_type(symbol, Variable)

        if type(symbol) == Window:
            return symbol.value.prop(self[1])

        else:
            print self
            raise Exception


class Return(List):
    def to_cpp(self, env=defaultdict(bool)):
        return 'return %s;' % self[0].to_cpp(env)
    def evaluate(self, env):
        raise InterpreterReturn(self[0].evaluate(env))

class Break(List):
    def to_cpp(self, env=defaultdict(bool)):
        return 'break;'
    def evaluate(self, env):
        raise InterpreterBreak

class If(List):
    def to_cpp(self, env=defaultdict(bool)):
        if len(self) == 2: # simple if (condition) {statement}
          return 'if (%s) %s' % cpp_tuple(self, env)
        elif len(self) == 3: # full if (condition) {statement} else {statement}
          return 'if (%s) %s else %s' % cpp_tuple(self, env)
        else:
          raise InternalException("Wrong type of if statement with %s length" % len(self))
    def evaluate(self, env):
        if len(self) == 2: # simple if (condition) {statement}
            condition, statement = self
            if condition.evaluate(env):
                statement.evaluate(env)
        elif len(self) == 3: # full if (condition) {statement} else {statement}
            condition, if_statement, else_statement
            if condition.evaluate(env):
                if_statement.evaluate(env)
            else:
                else_statement.evaluate(env)

        else:
          raise InternalException("Wrong type of if statement with %s length" % len(self))


class While(List):
    def to_cpp(self, env=defaultdict(bool)):
        return 'while (%s) %s' % cpp_tuple(self, env)
    def evaluate(self, env):
        expression, statement = self
        while (True):
            result = expression.evaluate(env)
            if not result:
                break
            try:
                statement.evaluate(env)
            except InterpreterBreak:
                break

class Read(List): pass
class Write(List): pass


class Random(List):
    def to_cpp(self, env=defaultdict(bool)):
        if len(self) == 2:
            min_limit, max_limit = cpp_tuple(self[0:2])
        else:
            min_limit, max_limit = ('0', 'INT_MAX')

        return '0 /* Random not implemented yet */'
        raise CompilerException('Random not implemented yet')
        #if env['@in_parallel_context']:
        #    return random_template % { 'min_value' : min_limit,
        #                               'max_value' : max_limit }
    def evaluate(self, env):
        if len(self) == 2:
            min_limit, max_limit = self
        else:
            min_limit = 0
            max_limit = (2**32)/2-1


        if not '@output' in env:
            print 'Warning, no output variable, can\'t perform rand()'
            return

        output = env['@output']

        expand_to_range = ElementwiseKernel(
                arguments="float min, float max, {output_type} *output, float *input".format(output_type=dtype_to_ctype[output.dtype]),
                operation="output[i] = input[i]*(max-min)+min",
                name="expand_to_range")

        expand_to_range(min_limit, max_limit, output, pycuda.curandom.rand(output.shape))

        return output

reduce_template = """\
thrust::reduce({input_data}.thrustPointer(), {input_data}.thrustEndPointer())\
"""
class ParallelReduce(List):
    def to_cpp(self, env=defaultdict(bool)):
        function = None

        if len(self) == 1:
            input = self[0]
        elif len(self) == 2:
            input, function = self
        else:
            raise InternalException("Wrong list length to parallel reduce")

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
            return reduce_template.format(input_data=input.name)

    def evaluate(self, env, output=None):
        function = None
        if len(self) == 1:
            input = self[0]
        elif len(self) == 2:
            input, function = self
        else:
            raise InternalException("Wrong list length to parallel reduce")


        input = env[input]
        reduction = ReductionKernel(input.dtype, neutral='0',
                                    reduce_expr='a+b', map_expr='in[i]',
                                    arguments='{type} *in'.format(type=dtype_to_ctype[input.dtype]))

        #return numpy.add.reduce(input.value.reshape(input.width*input.height))
        return reduction(input).get()

sort_template = """
// Sorting '%(input_data)s' and placing it into '%(output_data)s'

// Thrust then performs the hard work, and we swap pointers at the end if needed
{
  %(input_data)s.copyTo(%(output_data)s); //TODO: Check if pointers are the same... and don't copy
  thrust::sort(%(output_data)s.thrustPointer(), %(output_data)s.thrustEndPointer());
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
                                     'output_data' : output.name }

    def evaluate(self, env, output=None):
        if len(self) == 1:
            input, function = self[0], None
        elif len(self) == 2:
            input, function = self
        else:
            raise InternalException("Wrong list length to parallel sort")

        input = env[input]

        #TODO: Actually use function
        if function:
            print('Warning: functions in reduce() calls are not implemented yet')
            return

        temp = input.value.copy().reshape(input.height*input.width)
        temp.sort()
        return temp.reshape(input.height, input.width)


class ForeachParameter(List):
    def to_cpp(self, env=defaultdict(bool)):
        print self

class ForeachInputParameter(ForeachParameter): pass
class ForeachOutputParameter(ForeachParameter): pass

parallel_context_declaration = """\
struct %(function_name)s {
    %(struct_members)s
    %(function_name)s(%(function_parameters)s) %(struct_member_initializations)s {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple _t) %(function_body)s
};
"""
parallel_context_call = """\
{
    FunctionIterator iterator = makeStartIterator(%(size_name)s.width, %(size_name)s.height);

    %(temp_arrays)s

    thrust::for_each(iterator, iterator+%(size_name)s.length(), %(function)s(%(variables)s));

    %(swaps)s
    %(releases)s
}
"""
class ParallelContext(List):
    def to_cpp(self, env=defaultdict(bool)):
        parameters, statements = self
        context_name = '_foreach_context_' + next_open_name()
        arrays = [] # allows us to check if we're already using this array under another alias
        outputs = []
        requested_variables = [] # Allows for function calling

        struct_member_variables = []
        function_parameters = []
        struct_member_initializations = []

        symbolTable.createScope()
        for parameter in parameters:
            piece, data = parameter
            if type(parameter) == ForeachOutputParameter:
                outputs.append(piece)

            if data not in arrays:
                arrays.append(data)
            else:
                raise CompilerException("{name} already used in this foreach loop".format(name=data))

            data = symbolTable.lookup(data)
            if data.type not in data_types:
                raise CompilerException("%s %s should be an array" % (data.type, data.name))

            symbolTable.add(StreamVariable(name=piece, type=data_to_scalar[data.type], array=data, cpp_name='%s.center(_x, _y)' % piece))

            struct_member_variables.append('Array2d<%s> %s;' % (data_to_scalar[data.type], piece))
            function_parameters.append('Array2d<%s> _%s' % (data_to_scalar[data.type], piece))
            struct_member_initializations.append('%(name)s(_%(name)s)' % { 'name' : piece })
            requested_variables.append(data.name)


        if list_contains_type(statements, ParallelContext):
            raise CompilerException("Parallel Contexts (foreach loops) can not be nested")

        if list_contains_type(statements, Return):
            raise CompilerException("Parallel Contexts (foreach loops) can not contain return statments")

        for variable in collect_elements_from_list(statements, Variable):
            if variable not in symbolTable.currentScope and symbolTable.lookup(variable):
                # We've got a variable that needs to be passed into this function
                variable = symbolTable.lookup(variable)
                if variable.type in data_types:
                    struct_member_variables.append('Array2d<%s> %s;' % (data_to_scalar[variable.type], variable.name))
                    function_parameters.append('Array2d<%s> _%s' % (data_to_scalar[variable.type], variable.name))
                    struct_member_initializations.append('%(name)s(_%(name)s)' % { 'name' : variable.name })
                else:
                    struct_member_variables.append('%s %s;' % (data_to_scalar[variable.type], variable.name))
                    function_parameters.append('%s _%s' % (data_to_scalar[variable.type], variable.name))
                    struct_member_initializations.append('%(name)s(_%(name)s)' % { 'name' : variable.name})

                requested_variables.append(variable.name)

        for assignment in collect_elements_from_list(statements, Assignment):
            symbol = symbolTable.lookup(assignment[0])
            if symbol.name not in outputs:
                raise CompilerException("Error, trying to assign a value to '{name}'. If you want to write to this stream variable, designate it as an output variable (i.e. 'foreach {name}! in {array}')".format(name=symbol.name, array=symbol.array.name))


        if len(parameters) > 0:
            struct_member_variables = indent('\n'.join(struct_member_variables), indent_first_line=False)
            function_parameters = ', '.join(function_parameters)
            struct_member_initializations = ": " + ', '.join(struct_member_initializations)
        else:
            struct_member_variables = function_parameters = struct_member_initializations = ''



        body = '{\n' + indent('\n'.join(map(lambda s: s.to_cpp(), statements))) + '\n}'
        body = indent(indent(body, indent_first_line=False), indent_first_line=False) # indent twice

        environment = { 'function_name' : context_name,
                        'function_body' : body,
                        'struct_members' : struct_member_variables,
                        'function_parameters' : function_parameters,
                        'struct_member_initializations' : struct_member_initializations }

        function_declaration = parallel_context_declaration % environment


        # Function Call
        size = symbolTable.lookup(parameters[0][1]).name + '.size()'

        outputs = [symbolTable.lookup(output) for output in outputs]

        temp_array_names = ['temp_array_%s' % i for i in range(len(outputs))]
        temp_arrays = [create_data(output.array.type, name, output.array.size) for output, name in zip(outputs, temp_array_names)]
        swaps = map(lambda (a, b): swap_data(a, b.array.name), zip(temp_array_names, outputs))
        releases = [release_data(output.array.type, temp) for output, temp in zip(outputs, temp_array_names)]

        temp_mapping = {}
        for temp, output in zip(temp_array_names, outputs):
            temp_mapping[output.array.name] = temp

        requested_variables = [temp_mapping[var] if var in temp_mapping else var for var in requested_variables]

        symbolTable.removeScope()

        function_call = parallel_context_call % { 'variables' : ', '.join(requested_variables),
                                         'temp_arrays' : indent(''.join(temp_arrays), indent_first_line=False),
                                         'swaps' : indent(''.join(swaps), indent_first_line=False),
                                         'releases' : indent(''.join(releases), indent_first_line=False),
                                         'function' : context_name,
                                         'size_name' : size }


        symbolTable.parallelContexts.append(function_declaration)
        return function_call

    def evaluate(self, env, output):
        name = self[0]
        function = env[name]

        arguments = map(lambda arg: arg.evaluate(env), self[1])

        array = output.array.copy()

        for y in xrange(output.height):
            for x in xrange(output.width):
                args = map(lambda arg: RuntimeWindow(x, y, arg), arguments)

                try:
                    value = function.node.run(args, env)
                except InterpreterBreak:
                    raise InterpreterException('Error: caught break statement outside of a loop in function %s', function.name)
                except InterpreterReturn as return_value:
                    value = return_value[0]

                array[y, x] = value

        return array



