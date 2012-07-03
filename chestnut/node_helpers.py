from collections import defaultdict
from lepl import List

from .exceptions import CompilerException
from .nodes import FunctionCall

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
        if function.name == 'random':
            return True

    return False


# push and pop a symbol table scope when running this function
def scoped(function):
    def wrapper(self, *args, **kwargs):
        self.symbols.createScope()

        result = function(self, *args, **kwargs)

        self.symbols.removeScope()
        return result

    return wrapper

def parallel_scoped(function):
    def wrapper(self, *args, **kwargs):
        self.symbols.createParallelScope()

        result = function(self, *args, **kwargs)

        self.symbols.removeScope()
        return result

    return wrapper


def check_is_symbol(name, environment, context=None):
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
    if (not leftData.width == rightData.width) or (not leftData.height == rightData.height) or (not leftData.depth == rightData.depth):
        raise CompilerException("Error, '{0}' ({0}.width, {0}.height, {0}.depth) and '{1}' ({1}.width, {1}.height, {1}.depth) have different dimensions.".format(leftData, rightData))


#helper to convert sub-members into strings
def cpp_tuple(obj, env=defaultdict(bool)):
    return tuple(map(lambda element: element.to_cpp(env), obj))

#helper to do nice code indenting
def indent(code, indent_first_line=True):
    if indent_first_line:
        code = '  ' + code
    return code.replace('\n', '\n  ')
