from lepl import *
from collections import namedtuple, defaultdict
from templates import *
from symboltable import SymbolTable, Scope, Keyword, Variable, Window, Data, ParallelFunction, SequentialFunction, DisplayWindow
import random, numpy

symbolTable = SymbolTable()

def check_is_symbol(name, environment=symbolTable):
    if not environment.lookup(name):
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


def extract_line_info(function):
    def wrapper(obj, env=defaultdict(bool)):
        node_info = obj # obj[0:-2]
        start_line, end_line = (0, 0) #obj[-2:]
        return function(obj, node_info, start_line, end_line, env)
    return wrapper


#helpers so that we don't get errors about undefined to_cpp methods
class Symbol(str):
    def to_cpp(self, env=None):
        check_is_symbol(self)
        return self
    def evaluate(self, env):
        symbol = env.lookup(self)
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
        self._value = bool(value)
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
        return '%s = %s' % cpp_tuple(self, env)
    def evaluate(self, env):
        symbol = env.lookup(self[0])
        symbol.value = self[1].evaluate(env)
        return symbol.value
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
    def evaluate(self, env):
        type, name = self[0], self[1]
        var = env.add(Variable(name, type))
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
      env.add(Data(name, type_, 0, 0)) #TODO: fix to not have 0,0 for uninitialized

    elif len(self) == 3: # adds a size
      type_, name, size = self
      env.add(Data(name, type_, size.width, size.height))

    elif len(self) == 4: # adds an initialization
      type_, name, size, initialization = self
      symbol = env.add(Data(name, type_, size.width, size.height))

      symbol.value = initialization.evaluate(env)

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
        env.add(SequentialFunction(name, type_, parameters, ok_for_device=True, node=self))

    def run(self, arguments, env):
        type_, name, parameters, block = self
        env.createScope()

        for parameter, argument in zip(parameters, arguments):
            symbol = env.add(Variable(name=parameter.name, type=parameter.type))
            symbol.value = argument # argument was evaluated before the function call 

        result = block.evaluate(env)

        env.removeScope()
        return result

class ParallelFunctionDeclaration(List):
    def to_cpp(self, env=defaultdict(bool)):
        type_, name, parameters, block = self
        symbolTable.add(ParallelFunction(name, parameters, node=self))
        symbolTable.createScope()

        windowCount = 0
        for parameter in parameters:
            if parameter.type == 'window':
                symbolTable.add(Window(name=parameter.name, number=windowCount))
                windowCount += 1
            else:
                symbolTable.add(Variable(name=parameter.name, type=parameter.type))

        device_function = create_device_function(self)

        symbolTable.removeScope()
        return device_function

    def evaluate(self, env):
        type_, name, parameters, block = self
        env.add(ParallelFunction(name, parameters, node=self))

    def run(self, arguments, env):
        name, parameters, block = self
        env.createScope()

        for parameter, argument in zip(parameters, arguments):
            symbol = env.add(Variable(name=parameter.name, type=parameter.type))
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
        env.createScope()
        for statement in self:
            statement.evaluate(env)
        env.removeScope()


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
  thrust::device_vector<int> *unpadded = %(name)s.unpadded();
  thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(_%(name)s_display.displayData(), unpadded->begin())),
                   thrust::make_zip_iterator(thrust::make_tuple(_%(name)s_display.displayData()+%(size)s, unpadded->begin()+%(size)s)),
                  _chestnut_green_color_conversion_functor());
}
_%(name)s_display.updateGL();
_app.processEvents();
"""
class DataDisplay(List):
    def to_cpp(self, env=defaultdict(bool)):
        data = symbolTable.lookup(self[0])

        display_env = { 'name' : data.name,
                        'width' : data.width,
                        'height' : data.height,
                        'size' : data.size }

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
  hostData = *%(data)s.mainData;

  printArray2D(hostData, %(width)s, %(height)s, %(padding)s);
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
                                  'padding' : 1,
                                  'length' : pad(data.width)*pad(data.height) }
    def evaluate(self, env):
        data = self[0]
        data = env.lookup(data)

        print data.value

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

        #check_is_symbol(function)
        function = env.lookup(function)
        #check_type(function, SequentialFunction)

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
        'topLeft' : r'thrust::get<0>(thrust::get<%(window_num)s>(thrust::get<1>(t))) /* <- topLeft */',
        'top' : r'thrust::get<1>(thrust::get<%(window_num)s>(thrust::get<1>(t))) /* <- top */',
        'topRight' : r'thrust::get<2>(thrust::get<%(window_num)s>(thrust::get<1>(t))) /* <- topRight */',

        'left' : r'thrust::get<3>(thrust::get<%(window_num)s>(thrust::get<1>(t))) /* <- left */',
        'center' : r'thrust::get<4>(thrust::get<%(window_num)s>(thrust::get<1>(t))) /* <- center */',
        'right' : r'thrust::get<5>(thrust::get<%(window_num)s>(thrust::get<1>(t))) /* <- right */',

        'bottomLeft' : r'thrust::get<6>(thrust::get<%(window_num)s>(thrust::get<1>(t))) /* <- bottomLeft */',
        'bottom' : r'thrust::get<7>(thrust::get<%(window_num)s>(thrust::get<1>(t))) /* <- bottom */',
        'bottomRight' : r'thrust::get<8>(thrust::get<%(window_num)s>(thrust::get<1>(t)) )/* <- bottomRight */',

        'x' : r'(thrust::get<2>(t) %% thrust::get<3>(t) - 1) /* <- x */',
        'y' : r'(thrust::get<2>(t) / thrust::get<3>(t) - 1) /* <- y */',
        'width' : r'(thrust::get<3>(t) - 2) /* <- width */',
        'height' : r'(thrust::get<4>(t) - 2) /* <- height */',
               }

class Property(List):
    def to_cpp(self, env=defaultdict(bool)):
        name, property_ = self
        check_is_symbol(name)
        symbol = symbolTable.lookup(name)


        #check_type(symbol, Variable) #TODO: Support Data properties


        #TODO: Pull in support for window nums other than 0
        if type(symbol) == Window:
            return coordinates[self[1]] % { 'window_num' : symbol.number }
        else:
            print symbolTable
            print self
            raise Exception

    def evaluate(self, env):
        check_is_symbol(self[0], env)
        symbol = env.lookup(self[0])
        #check_type(symbol, Variable)

        if type(symbol) == Window:
            return symbol.value.prop(self[1])

        else:
            print self
            raise Exception


class Return(List):
    def to_cpp(self, env=defaultdict(bool)):
        if env['sequential']:
            return 'return %s;' % self[0].to_cpp(env)
        else:
            return '{ // Returning from device function requires 2 steps\n' + indent('thrust::get<0>(t) = %s;\nreturn;' % self[0].to_cpp(env)) + '\n}'
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


class ParallelAssignment(List):
    def to_cpp(self, env=defaultdict(bool)):
        env = defaultdict(bool, data_to_assign=self[0])
        return self[1].to_cpp(env)

    def evaluate(self, env):
        name = self[0]
        data = env.lookup(name)

        if type(data) == Data:
            data.value = self[1].evaluate(env, data)
        elif type(data) == Variable:
            data = self[1].evaluate(env, data)



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
    def evaluate(self, env, output):
        if len(self) == 2:
            min_limit, max_limit = self
            max_limit -= 1
        else:
            min_limit = 0
            max_limit = (2**32)/2-1


        for y in xrange(output.height):
            for x in xrange(output.width):
                output.setAt(x, y, random.randint(min_limit, max_limit))

        return output.value

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

    def evaluate(self, env, output=None):
        function = None
        if len(self) == 1:
            input = self[0]
        elif len(self) == 2:
            input, function = self
        else:
            raise InternalException("Wrong list length to parallel reduce")

        input = env.lookup(input)
        return numpy.add.reduce(input.value.reshape(input.width*input.height))

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

    def evaluate(self, env, output=None):
        if len(self) == 1:
            input, function = self[0], None
        elif len(self) == 2:
            input, function = self
        else:
            raise InternalException("Wrong list length to parallel sort")

        input = env.lookup(input)

        #TODO: Actually use function
        if function:
            print('Warning: functions in reduce() calls are not implemented yet')
            return

        temp = input.value.copy().reshape(input.height*input.width)
        temp.sort()
        return temp.reshape(input.height, input.width)


map_template = """
{
    %(maybe_wraps)s

    ChestnutDetail<%(type)s>::Kernel%(num_inputs)sWindowIter startIterator
        = ChestnutDetail<%(type)s>::startIterator(%(output_data)s%(input_datas)s);
    ChestnutDetail<%(type)s>::Kernel%(num_inputs)sWindowIter endIterator
        = ChestnutDetail<%(type)s>::endIterator(%(output_data)s%(input_datas)s);


    thrust::for_each(startIterator, endIterator, %(function)s_functor(%(variable_arguments)s));

    // Always swap the output data b/c we are putting data into the secondary array
    %(output_data)s.swap();
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
            if parameter.type == 'window':
                data_arguments.append(argument.to_cpp(env))
            else:
                variable_arguments.append(argument.to_cpp(env))


        supported_window_sizes = [0, 1, 2, 3, 4]
        if len(data_arguments) not in supported_window_sizes:
            raise CompilerException('Error, the number of Data Windows is %s but chestnut only supports windows of sizes %s' % (len(data_arguments), supported_window_sizes))

        return map_template % { 'input_datas' : ''.join(map(lambda arg: ', ' + arg, data_arguments)),
                                'output_data' : output.name,
                                'num_inputs' : len(data_arguments),
                                'maybe_wraps' : '\n'.join(map(lambda data: '%s.maybeWrap();' % data, data_arguments)),
                                'variable_arguments' : ','.join(variable_arguments),
                                'type' : type_map[output.type],
                                'function' : function.name }

    def evaluate(self, env, output):
        name = self[0]
        function = env.lookup(name)

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

