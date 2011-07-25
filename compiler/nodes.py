from lepl import *
from collections import namedtuple, defaultdict
from templates import *
from symboltable import scalar_types, data_types
from symboltable import SymbolTable, Scope, Keyword, StreamVariable, Variable, Window, Data, ParallelFunction, SequentialFunction, NeutralFunction, DisplayWindow
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

class Context(List):
    @property
    def start_line(self):
        return self[0]
    @property
    def end_line(self):
        return self[1]

class ChestnutNode(List):
    def __init__(self, items):
        super(ChestnutNode, self).__init__(items)

        if len(self) == 0 or type(self[-1]) != Context:
            self.extend([Context([None, None])])
    @property
    def context(self):
        if type(self[-1]) == Context:
            return self[-1]
        return None


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


def list_contains_function(deep_list, function_name):
    for function in collect_elements_from_list(deep_list, FunctionCall):
        if len(function) > 0 and function[0] == 'random':
            return True

    return False

def next_open_name():
    global current_name_number
    current_name_number += 1
    return str(current_name_number)


def safe_index(function):
    def wrapper(*args, **kwargs):
        try:
            property = function(*args, **kwargs)
            if type(property) == Context:
                return None
            return property
        except IndexError:
            return None
    return wrapper

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



def check_is_symbol(name, environment=symbolTable, context=None):
    if not environment.lookup(name):
        raise CompilerException("Error, the symbol '%s' was used but it hasn't been declared yet" % name, context)

def check_type(symbol, *requiredTypes):
    if type(symbol) not in requiredTypes:
        if len(requiredTypes) == 1:
            raise CompilerException("Error, the symbol '%s' is a %s, but a symbol of type %s was expected" % \
                    (symbol.name, type(symbol), requiredTypes[0]))
        else:
            raise CompilerException("Error, the symbol '%s' is a %s, but one of %s was expected" % \
                    (symbol.name, type(symbol), requiredTypes))


def check_dimensions_are_equal(leftData, rightData):
    assert type(leftData) == type(rightData) == Data
    if (not leftData.width == rightData.width) or (not leftData.height == rightData.height):
        raise CompilerException("Error, '%s' (%s, %s) and '%s' (%s, %s) have different dimensions." % \
         (rightData.name, rightData.width, rightData.height, leftData.name, leftData.width, leftData.height))


class Type(str):
    def to_cpp(self, env=None):
        return type_map[self]
    def evaluate(self, env):
        return

#helpers so that we don't get errors about undefined to_cpp methods
class Symbol(str):
    def to_cpp(self, env=None):
        check_is_symbol(self)
        return symbolTable.lookup(self).cpp_name
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
class CompilerException(Exception):
    def __init__(self, message, node=None):
        self.message = message
        self.node = node

    def __str__(self):
        if not self.node or not self.node.context:
            return "Error\n  {message}".format(message=self.message)
        elif self.node.context.start_line == self.node.context.end_line:
            return "Error around line {line_num}\n  {message}".format(line_num=self.node.context.start_line,
                                                                            message=self.message)
        else:
            return "Error around lines {start} and {end}\n  {message}".format(start=self.node.context.start_line,
                                                                            end=self.node.context.end_line,
                                                                            message=self.message)


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

### Other Operators ###
class Mod(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "((int)%s %% %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return self[0].evaluate(env) % self[1].evaluate(env)


### Comparetors ###
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
        env = defaultdict(bool, env, data_to_assign=output)
        return '%s = %s' % (output.to_cpp(env), expression.to_cpp(env))
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

class VariableDeclaration(ChestnutNode):
    @property
    def type(self): return self[0]

    @property
    def name(self): return self[1]

    @property
    @safe_index
    def initialization(self): return self[2]

    def to_cpp(self, env=defaultdict(bool)):
        symbolTable.add(Variable(self.name, self.type))

        if self.initialization:
            env = defaultdict(bool, env, variable_to_assign=self.name)
            return '%s %s;\n%s' % cpp_tuple([self.type, self.name, self.initialization], env)
        else:
            return '%s %s;' % cpp_tuple([self.type, self.name], env)
    def evaluate(self, env):
        var = env[self.name] = Variable(self.name, self.type)

        if self.initialization:
            var.value = self.initialization.evaluate(env)


class DataDeclaration(ChestnutNode):
  def to_cpp(self, env=defaultdict(bool)):
      type_, name, size, context = self
      symbolTable.add(Data(name, type_, size.width, size.height))
      return create_data(type_, name, size)

  def evaluate(self, env):
      type_, name, size, context = self
      (size.width, size.height)
      numpy_type_map[type_]
      env['@device_memory_pool'].allocate
      env[name] = DeviceArray((size.width, size.height), dtype=numpy_type_map[type_], allocator=env['@device_memory_pool'].allocate)


object_template = """\
struct {name} {{
{member_variables}
{functions}
}};
"""
class ObjectDeclaration(ChestnutNode):
    def to_cpp(self, env=defaultdict(bool)):
        object_name, members, context = self

        member_variables = []

        for type, name in members:
            member_variables.append('%s %s;' % (chestnut_to_c[type], name))

        symbolTable.add(Object(name=object_name, members=members))
        symbolTable.structures.append(object_template.format(name=object_name,
                                                             member_variables=indent('\n'.join(member_variables)),
                                                             functions=''))
        return ''

host_function_template = """\
__host__ %(type)s %(function_name)s(%(parameters)s) %(block)s
"""
class SequentialFunctionDeclaration(ChestnutNode):
    def to_cpp(self, env=defaultdict(bool)):
        type_, name, parameters, block, context = self

        symbolTable.add(SequentialFunction(name, type_, parameters, node=self))
        symbolTable.createScope()
        for parameter in parameters:
            symbolTable.add(Variable(name=parameter.name, type=parameter.type))

        environment = { 'function_name' : name,
                        'type' : type_,
                        'parameters' : ', '.join(map(lambda type, name: '%s %s' % (type, name), tuple(parameters))),
                        'block' : block.to_cpp(env) }

        host_function = host_function_template % environment

        symbolTable.removeScope()
        return host_function

    def evaluate(self, env):
        type_, name, parameters, block, context = self
        env[name] = SequentialFunction(name, type_, parameters, ok_for_device=True, node=self)

    def run(self, arguments, env):
        type_, name, parameters, block, context = self

        function_scope = Scope(env)

        for parameter, argument in zip(parameters, arguments):
            symbol = function_scope.add(Variable(name=parameter.name, type=parameter.type))
            symbol.value = argument # argument was evaluated before the function call 

        result = block.evaluate(function_scope)

        return result

device_function_template = """\
__device__ %(type)s %(function_name)s(%(parameters)s) %(block)s
"""
class ParallelFunctionDeclaration(ChestnutNode):
    def to_cpp(self, env=defaultdict(bool)):
        type_, name, parameters, block, context = self

        if list_contains_type(block, ParallelContext):
            raise CompilerException("Parallel contexts (foreach loops) can not exist inside of parallel functions", self)

        symbolTable.add(ParallelFunction(name, type_, parameters, node=self))
        symbolTable.createScope()
        env['@in_parallel_context'] = True

        for parameter in parameters:
            symbolTable.add(Variable(name=parameter.name, type=parameter.type))

        environment = { 'function_name' : name,
                        'type' : chestnut_to_c[type_],
                        'parameters' : ', '.join(map(lambda (type, name): '%s %s' % (chestnut_to_c[type], name), tuple(parameters))),
                        'block' : block.to_cpp(env) }

        device_function = device_function_template % environment


        del env['@in_parallel_context']
        symbolTable.removeScope()
        return device_function

    def evaluate(self, env):
        type_, name, parameters, block, context = self
        env[name] = ParallelFunction(name, type_, parameters, node=self)

    def run(self, arguments, env):
        name, parameters, block, context = self

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
    def to_cpp(self, env):
        symbolTable.createScope()
        cpp = '{\n' + indent('\n'.join(cpp_tuple(self, env))) + '\n}'
        symbolTable.removeScope()
        return cpp
    def evaluate(self, env):
        block_scope = Scope(env)
        for statement in self:
            statement.evaluate(block_scope)


class VariableInitialization(ChestnutNode):
    def to_cpp(self, env=defaultdict(bool)):
        return '%s = %s;' % (env['variable_to_assign'], self[0].to_cpp(env))
    def evaluate(self, env):
        return self[0].evaluate(env)

class DataInitialization(ChestnutNode):
    def to_cpp(self, env=defaultdict(bool)):
        return self[0].to_cpp(env)
    def evaluate(self, env):
        return self[0].evaluate(env)


display_functor_template = """\
template <typename InputType>
struct _color_convert__{display_function_name} : public thrust::unary_function<{input_data_type}, Color> {{
    __device__
    Color operator()({input_data_type} value) {{
        return {display_function_name}(value);
    }}
}};
"""
display_template = """\
{
  thrust::copy(thrust::make_transform_iterator(%(input)s.thrustPointer(), %(display_function)s<%(template_types)s>()),
               thrust::make_transform_iterator(%(input)s.thrustEndPointer(), %(display_function)s<%(template_types)s>()),
               _%(input)s_display.displayData());
}
_%(input)s_display.updateGL();
_app.processEvents();
"""
class DataDisplay(ChestnutNode):
    @property
    def data(self): return self[0]

    @property
    @safe_index
    def display_function(self): return self[1]

    def to_cpp(self, env=defaultdict(bool)):
        data = symbolTable.lookup(self.data)

        if self.display_function:
            display_function = symbolTable.lookup(self.display_function)
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


            functor = display_functor_template.format(display_function_name=display_function.name,
                                                      input_data_type=chestnut_to_c[data.type])

            symbolTable.parallelContexts.append(functor)

            display_function = '_color_convert__{}'.format(display_function.name)
        else:
            display_function = '_chestnut_default_color_conversion'

        template_types = '%s' % type_map[data.type]

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
class DataPrint(ChestnutNode):
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

class Print(ChestnutNode):
    def to_cpp(self, env=defaultdict(bool)):
        num_format_placeholders = self[0].to_cpp(env).count('%s')
        num_args = len(self)-2

        if num_format_placeholders != num_args:
            raise CompilerException("There are %s format specifiers but %s arguments to print()" % \
                    (num_format_placeholders, num_args), self)
        format_substrings = self[0].to_cpp(env).split('%s')

        code = 'std::cout'
        for substring, placeholder in zip(format_substrings[:-1], cpp_tuple(self[1:-1], env)):
            code += ' << "%s"' % substring
            code += ' << %s' % placeholder
        code += ' << "%s" << std::endl' % format_substrings[-1]

        return code
    def evaluate(self, env):
        text = self[0]
        args = map(lambda arg: arg.evaluate(env), self[1:-1])

        print(text % tuple(args))


sequential_function_call_template = """\
%(function_name)s(%(arguments)s)\
"""
class SequentialFunctionCall(ChestnutNode):
    def to_cpp(self, env=defaultdict(bool)):
        function = self[0]
        arguments = self[1]

        #TODO: Make this generic instead of a hack
        if function == 'sort':
            return DataSort(arguments).to_cpp(env)
        elif function == 'reduce':
            return DataReduce(arguments).to_cpp(env)
        elif function == 'display':
            return DataDisplay(arguments).to_cpp(env)
        elif function == 'print':
            if symbolTable.lookup(arguments[0]) and type(symbolTable.lookup(arguments[0])) == Data:
                return DataPrint(arguments).to_cpp(env)
            else:
                return Print(arguments).to_cpp(env)
        elif function == 'random':
            return Random(arguments).to_cpp(env)

        check_is_symbol(function)
        function = symbolTable.lookup(function)
        check_type(function, SequentialFunction, NeutralFunction)

        if not len(arguments) == len(function.parameters):
            print arguments
            raise CompilerException("Error, sequential function ':%s' takes %s parameters but %s were given" % \
                    (function.name, len(function.parameters), len(arguments)))


        return sequential_function_call_template % { 'function_name' : function.name,
                                                     'arguments' : ', '.join(cpp_tuple(arguments, env)) }

    def evaluate(self, env):
        function = self[0]
        arguments = map(lambda obj: obj.evaluate(env), self[1])

        function = env[function]

        try:
            function.node.run(arguments, env)
        except InterpreterBreak:
            raise InterpreterException('Error: caught break statement outside of a loop in function %s', function.name)
        except InterpreterReturn as return_value:
            return return_value[0]


parallel_function_call_template = """\
%(function_name)s(%(arguments)s)\
"""
class ParallelFunctionCall(ChestnutNode):
    def to_cpp(self, env=defaultdict(bool)):
        function = self[0]
        arguments = self[1]


        #TODO: Make this generic instead of a hack
        if function == 'location':
            return ParallelLocation(arguments).to_cpp(env)
        elif function == 'window':
            return ParallelWindow(arguments).to_cpp(env)
        elif function == 'random':
            return Random(arguments).to_cpp(env)

        try:
            check_is_symbol(function)
        except CompilerException:
            raise CompilerException("Can't find the function called '%s', is this a misspelling?" % function, self)
        function = symbolTable.lookup(function)
        check_type(function, ParallelFunction, NeutralFunction)

        if not len(arguments) == len(function.parameters):
            raise CompilerException("The parallel function '%s' takes %s parameters but %s were given" % \
                    (function.name, len(function.parameters), len(arguments)), self)


        return parallel_function_call_template % { 'function_name' : function.name,
                                                   'arguments' : ', '.join(cpp_tuple(arguments, env)) }

    def evaluate(self, env):
        function = self[0]
        arguments = map(lambda obj: obj.evaluate(env), self[1])

        function = env[function]

        try:
            function.node.run(arguments, env)
        except InterpreterBreak:
            raise InterpreterException('Error: caught break statement outside of a loop in function %s', function.name)
        except InterpreterReturn as return_value:
            return return_value[0]

class FunctionCall(ChestnutNode):
    def to_cpp(self, env=defaultdict(bool)):
        if env['@in_parallel_context']:
            return ParallelFunctionCall(self).to_cpp(env)
        else:
            return SequentialFunctionCall(self).to_cpp(env)

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
        'opacity' : 'w' }

property_template = """\
%(name)s.%(property)s(%(parameters)s)\
"""
class Property(List):
    def to_cpp(self, env=defaultdict(bool)):
        expr, rest = self[0], self[1:]

        # New property pseudo code
        property_cpp = expr.to_cpp(env)

        for member_identifier in rest:
            #check_that_symbol_type_has_member(current_symbol_type, member_identifier) # need to implement
            #current_symbol_type = get_return_type_for_member(current_symbol_type, member_identifier) # need to implement

            property_cpp = property_template % { 'name' : property_cpp,
                                                 'property' : member_identifier,
                                                 'parameters' : '' }

        return property_cpp

        #if type(symbol) == Variable:
        #    if symbol.type == 'Color':
        #        return '%s.%s' % (symbol.name, color_properties[property_])
        #    elif symbol.type == 'Point2d' and property_ in ['x', 'y']:
        #        return '%s.%s' % (symbol.name, property_)
        #    elif symbol.type == 'Size2d' and property_ in ['width', 'height']:
        #        return '%s.%s' % (symbol.name, property_)
        #    elif symbol.type in ['IntWindow2d', 'RealWindow2d', 'ColorWindow2d'] \
        #       and property_ in ['topLeft', 'top', 'topRight', 'left', 'center', 'right', 'bottomLeft', 'bottom', 'bottomRight']:
        #        return property_template % { 'name' : name,
        #                                    'property' : property_,
        #                                    'parameters' : '' }
        #    else:
        #        raise Exception('unknown property variable type %s.%s' % (symbol.type, property_))

        #elif type(symbol) == Data:
        #    if property_ not in ['size']:
        #        raise CompilerException("Error, property '%s' not part of the data '%s'" % (property_, name))
        #    return property_template % { 'name' : name,
        #                                 'property' : property_,
        #                                 'parameters' : '' }

        #else:
        #    print symbolTable
        #    print self
        #    raise Exception('unknown property type')

    def evaluate(self, env):
        check_is_symbol(self[0], env)
        symbol = env[self[0]]
        #check_type(symbol, Variable)

        if type(symbol) == Window:
            return symbol.value.prop(self[1])

        else:
            print self
            raise Exception


class Return(ChestnutNode):
    def to_cpp(self, env=defaultdict(bool)):
        return 'return %s;' % self[0].to_cpp(env)
    def evaluate(self, env):
        raise InterpreterReturn(self[0].evaluate(env))

class Break(ChestnutNode):
    def to_cpp(self, env=defaultdict(bool)):
        return 'break;'
    def evaluate(self, env):
        raise InterpreterBreak

class If(ChestnutNode):
    @property
    def condition(self): return self[0]

    @property
    def if_statement(self): return self[1]

    @property
    @safe_index
    def else_statement(self) : return self[2]

    def to_cpp(self, env=defaultdict(bool)):
        if not self.else_statement:
            return 'if (%s) %s' % cpp_tuple([self.condition, self.if_statement], env)
        else:
            return 'if (%s) %s else %s' % cpp_tuple([self.condition, self.if_statement, self.else_statement], env)

    def evaluate(self, env):
        if not self.else_statement:
            if self.condition.evaluate(env):
                self.if_statement.evaluate(env)
        else:
            if self.condition.evaluate(env):
                self.if_statement.evaluate(env)
            else:
                self.else_statement.evaluate(env)

class While(ChestnutNode):
    @property
    def condition(self): return self[0]

    @property
    @safe_index
    def statement(self): return self[1]

    def to_cpp(self, env=defaultdict(bool)):
        return 'while (%s) %s' % cpp_tuple([self.condition, self.statement], env)

    def evaluate(self, env):
        while (True):
            result = self.expression.evaluate(env)
            if not result:
                break
            try:
                self.statement.evaluate(env)
            except InterpreterBreak:
                break

class Read(ChestnutNode): pass
class Write(ChestnutNode): pass

random_device_template = """walnut_random()"""
class Random(ChestnutNode):
    def to_cpp(self, env=defaultdict(bool)):
        if env['@in_parallel_context']:
            return random_device_template
        else:
            raise CompilerException("Sequential Random not implemented")

    def evaluate(self, env):
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


class ParallelLocation(ChestnutNode):
    def to_cpp(self, env=defaultdict(bool)):
        if type(symbolTable.lookup(self[0])) is not StreamVariable:
            raise CompilerException("Trying to take the location of a normal (non-stream) variable '%s'. This doesn't work" % self[0])
        return 'Point2d(_x, _y)'

class ParallelWindow(ChestnutNode):
    def to_cpp(self, env=defaultdict(bool)):
        symbol = symbolTable.lookup(self[0])

        if type(symbol) is not StreamVariable:
            raise CompilerException("Trying to take the window of a normal (non-stream) variable '%s'. This doesn't work" % self[0])

        return '%s(%s, _x, _y)' % (chestnut_to_c[datatype_to_windowtype[symbol.array.type]], symbol.array.name)

reduce_template = """\
thrust::reduce({input_data}.thrustPointer(), {input_data}.thrustEndPointer())\
"""
class DataReduce(ChestnutNode):
    @property
    def input(self): return self[0]

    @property
    @safe_index
    def function(self): return self[1]

    def to_cpp(self, env=defaultdict(bool)):
        function = None

        check_is_symbol(self.input)
        input = symbolTable.lookup(self.input)

        check_type(input, Data)

        #TODO: Actually use function
        if self.function:
            print('Warning: functions in reduce() calls are not implemented yet')
            check_is_symbol(self.function)
            function = symbolTable.lookup(self.function)
            check_type(self.function, SequentialFunction)
            return ''

        else:
            return reduce_template.format(input_data=input.name)

    def evaluate(self, env, output=None):
        input = env[self.input]
        reduction = ReductionKernel(input.dtype, neutral='0',
                                    reduce_expr='a+b', map_expr='in[i]',
                                    arguments='{type} *in'.format(type=dtype_to_ctype[input.dtype]))

        #return numpy.add.reduce(input.value.reshape(input.width*input.height))
        return reduction(input).get()

sort_template = """%(input_data)s.sorted_copy_in(%(output_data)s)"""
class DataSort(ChestnutNode):
    @property
    def input(self): return self[0]

    @property
    @safe_index
    def function(self): return self[1]

    def to_cpp(self, env=defaultdict(bool)):
        function = None
        output = env['data_to_assign']
        if not output:
            raise InternalException("The environment '%s' doesn't have a 'data_to_assign' variable set" % env)

        check_is_symbol(self.input)
        check_is_symbol(output)

        input = symbolTable.lookup(self.input)
        output = symbolTable.lookup(output)

        check_type(input, Data)
        check_type(output, Data)

        #TODO: Actually use function
        if self.function:
            print('Warning: functions in reduce() calls are not implemented yet')
            check_is_symbol(self.function)
            function = symbolTable.lookup(self.function)
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


class ForeachParameter(ChestnutNode):
    def to_cpp(self, env=defaultdict(bool)):
        print self

class ForeachParameter(ForeachParameter): pass

parallel_context_declaration = """\
struct %(function_name)s {
    %(struct_members)s
    %(function_name)s(%(function_parameters)s) %(struct_member_initializations)s {}

    template <typename Tuple>
    __device__
    void operator()(Tuple _t) %(function_body)s
};
"""
parallel_context_call = """\
{
    FunctionIterator iterator = makeStartIterator(%(size_name)s.width(), %(size_name)s.height());

    %(temp_arrays)s

    thrust::for_each(iterator, iterator+%(size_name)s.length(), %(function)s(%(variables)s));

    %(swaps)s
    %(releases)s
}
"""
class ParallelContext(ChestnutNode):
    def to_cpp(self, env=defaultdict(bool)):
        parameters, statements, context = self
        env['@in_parallel_context'] = True

        context_name = '_foreach_context_' + next_open_name()
        arrays = [] # allows us to check if we're already using this array under another alias
        outputs = []
        requested_variables = [] # Allows for function calling

        struct_member_variables = []
        function_parameters = []
        struct_member_initializations = []

        copies = ''

        symbolTable.createScope()
        for parameter in parameters:
            piece, data, context = parameter

            if data not in arrays:
                arrays.append(data)
            else:
                raise CompilerException("{name} already used in this foreach loop".format(name=data), self)

            data = symbolTable.lookup(data)
            if data.type not in data_types:
                raise CompilerException("%s %s should be an array" % (data.type, data.name), self)


            symbolTable.add(StreamVariable(name=piece, type=data_to_scalar[data.type], array=data, cpp_name='Window2d<%s>(%s, _x, _y).center()'
                            % (chestnut_to_c[data.type], data.name)))

            symbolTable.add(data)

            struct_member_variables.append('Array2d<%s> %s;' % (chestnut_to_c[data.type], data.name))
            function_parameters.append('Array2d<%s> _%s' % (chestnut_to_c[data.type], data.name))
            struct_member_initializations.append('%(name)s(_%(name)s)' % { 'name' : data.name })
            requested_variables.append(data.name)

        if list_contains_type(statements, ParallelContext):
            raise CompilerException("Parallel Contexts (foreach loops) can not be nested")

        if list_contains_type(statements, Return):
            raise CompilerException("Parallel Contexts (foreach loops) can not contain return statments")


        for variable in collect_elements_from_list(statements, Symbol):
            if variable not in symbolTable.currentScope and symbolTable.lookup(variable) and type(symbolTable.lookup(variable)) not in [ParallelFunction, SequentialFunction, NeutralFunction]:
                # We've got a variable that needs to be passed into this function
                variable = symbolTable.lookup(variable)
                if variable.type in data_types:
                    struct_member_variables.append('Array2d<%s> %s;' % (chestnut_to_c[variable.type], variable.name))
                    function_parameters.append('Array2d<%s> _%s' % (chestnut_to_c[variable.type], variable.name))
                    struct_member_initializations.append('%(name)s(_%(name)s)' % { 'name' : variable.name })
                else:
                    struct_member_variables.append('%s %s;' % (chestnut_to_c[variable.type], variable.name))
                    function_parameters.append('%s _%s' % (chestnut_to_c[variable.type], variable.name))
                    struct_member_initializations.append('%(name)s(_%(name)s)' % { 'name' : variable.name})

                requested_variables.append(variable.name)

        for assignment in collect_elements_from_list(statements, Assignment):
            symbol = symbolTable.lookup(assignment[0])
            if symbol and type(symbol) == StreamVariable and symbol not in outputs:
                outputs.append(symbol)

                struct_member_variables.append('Array2d<%s> _original_values_for_%s;' % (chestnut_to_c[symbol.type], symbol.array.name))
                function_parameters.append('Array2d<%s> __original_values_for_%s' % (chestnut_to_c[symbol.type], symbol.array.name))
                struct_member_initializations.append('_original_values_for_%(name)s(__original_values_for_%(name)s)' % { 'name' : symbol.array.name })

                copies += '%s = %s;\n' % (symbol.cpp_name, 'Window2d<%s>(_original_values_for_%s, _x, _y).center()' % (chestnut_to_c[symbol.type], symbol.array.name))


        new_names = []

        for (new_name, original_name) in [('_new_values_for_%s' % output.array.name, output.array.name) for output in outputs]:
            #Swap order of new and old so that it matches the kernel order. Weird, I know
            requested_variables[requested_variables.index(original_name)] = new_name
            requested_variables.append(original_name)
            new_names.append(new_name)

        temp_arrays = [create_data(output.array.type, name, output.array.size) for output, name in zip(outputs, new_names)]
        swaps = map(lambda (a, b): swap_data(a, b.array.name), zip(new_names, outputs))
        releases = [release_data(output.array.type, temp) for output, temp in zip(outputs, new_names)]

        if len(parameters) > 0:
            struct_member_variables = indent(indent('\n'.join(struct_member_variables), indent_first_line=False), indent_first_line=False)
            function_parameters = ', '.join(function_parameters)
            struct_member_initializations = ": " + ', '.join(struct_member_initializations)
        else:
            struct_member_variables = function_parameters = struct_member_initializations = ''

        if list_contains_function(statements, 'random'):
            random_initializer = "curandState __random_state;\ncurand_init(hash(_index), 0, 0, &__random_state);\n"
        else:
            random_initializer = ''

        body = '{\n' + copies + indent(random_initializer + '\n'.join(map(lambda s: s.to_cpp(env), statements))) + '\n}'
        body = indent(indent(body, indent_first_line=False), indent_first_line=False) # indent twice

        environment = { 'function_name' : context_name,
                        'function_body' : body,
                        'struct_members' : struct_member_variables,
                        'function_parameters' : function_parameters,
                        'struct_member_initializations' : struct_member_initializations }

        function_declaration = parallel_context_declaration % environment


        # Function Call
        size = symbolTable.lookup(parameters[0][1]).name + '.size()'


        symbolTable.removeScope()

        function_call = parallel_context_call % { 'variables' : ', '.join(requested_variables),
                                         'temp_arrays' : indent(''.join(temp_arrays), indent_first_line=False),
                                         'swaps' : indent(''.join(swaps), indent_first_line=False),
                                         'releases' : indent(''.join(releases), indent_first_line=False),
                                         'function' : context_name,
                                         'size_name' : size }


        symbolTable.parallelContexts.append(function_declaration)

        del env['@in_parallel_context']
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



